"""ceemdan.py -- CEEMDAN on top of GPUBatchEMD, with PyEMD's CEEMDAN interface (EMD-signal 1.10.0).

TWO CLASSES
    CEEMDAN          drop-in replacement for PyEMD.CEEMDAN: CEEMDAN(...)(S) / ceemdan(S) return a numpy array.
    GPUBatchCEEMDAN  the same engine; its __call__ returns (device array, n_errors).

CONSTRUCTOR -- the same as PyEMD's CEEMDAN:

    CEEMDAN(trials=100, epsilon=0.005, ext_EMD=None, parallel=True, **kwargs)

    trials         number of noise realisations
    epsilon        noise amplitude relative to the std of the current residue
    ext_EMD        an EMD engine to use instead of building one from the kwargs: an EMD / GPUBatchEMD, or a CPU
                   PyEMD.EMD (its settings are copied onto a GPU engine)
    parallel       accepted for compatibility; the trials always run as one GPU batch
    kwargs         noise_scale=1.0  range_thr=0.01  total_power_thr=0.05  beta_progress=True  seed=None
                   noise_kind="normal" ("normal" or "uniform")  max_imf=100 (bound on the number of stages)
                   processes=None (no effect on a GPU); everything else, INCLUDING range_thr / total_power_thr
                   when you pass them, is forwarded to the EMD engine exactly as PyEMD forwards it to EMD(**kwargs).
                   GPU-only: chunk_trials (trials per GPU batch; default: as many as fit in memory).

THE ALGORITHM, as PyEMD does it -- and in PyEMD's arithmetic
    S is divided by np.std(S).  `trials` noise rows (std `noise_scale`, or the `noise` array as given, absolute scale)
    are fully decomposed, each row's modes divided by np.std of its first IMF (beta_progress).  cIMF 1 is the trial
    mean of EMD(S + epsilon * mode_0)[0] (summed from 0.0 in trial order, then divided by `trials`).  Stage k takes
    local_mean = sum over trials, in order, of EMD(res + beta * mode_k, max_imf=1)[-1] / trials, with
    beta = epsilon * np.std(res); the new cIMF is res - local_mean.  Stops at PyEMD's end condition (np.max, np.min,
    np.sum on the host).  Output rows: the cIMFs, then S - sum(cIMFs), all multiplied back by std(S).
    Every std, sum and threshold test is numpy's own, every EMD is PyEMD's bit for bit (kernels.py), so with the same
    noise the result is PyEMD's bit for bit.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

try:
    from . import kernels as _K
    from .emd import EMD, EMDOverflowWarning, GPUBatchEMD, as_gpu_emd, raise_for_error  # noqa: F401
except ImportError:
    import kernels as _K
    from emd import EMD, EMDOverflowWarning, GPUBatchEMD, as_gpu_emd, raise_for_error  # noqa: F401

try:
    from tqdm import tqdm  # PyEMD depends on tqdm too; here it is optional
except ImportError:  # pragma: no cover
    tqdm = None


def _noise_backend(v):
    v = str(v).lower()
    if v not in ("numpy", "cupy"):
        raise ValueError(f"noise_backend must be 'numpy' (PyEMD's RandomState stream) or 'cupy' (GPU), got {v!r}")
    return v


def _make_rng(backend, seed):
    return np.random.RandomState(seed) if backend == "numpy" else cp.random.RandomState(seed)

# kwargs that belong to CEEMDAN only: PyEMD forwards them to EMD(**kwargs), which ignores them; here they are
# filtered out so that the EMD engine's unknown-keyword warning stays meaningful.
_CEEMDAN_ONLY_KWARGS = ("noise_scale", "beta_progress", "seed", "noise_kind", "max_imf", "processes", "noise_backend")


class GPUBatchCEEMDAN:
    noise_kinds_all = ["normal", "uniform"]
    _noise_backend_default = "cupy"

    def __init__(self, trials=100, epsilon=0.005, ext_EMD=None, parallel=True, **kwargs):
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        chunk_trials = kwargs.pop("chunk_trials", None)
        if chunk_trials is not None and chunk_trials < 1:
            raise ValueError(f"chunk_trials must be >= 1 or None, got {chunk_trials}")

        # ---- PyEMD.CEEMDAN.__init__ ----
        self.trials = int(trials)
        self.epsilon = float(epsilon)
        self.noise_scale = float(kwargs.get("noise_scale", 1.0))
        # The two thresholds below only feed the CEEMDAN end condition; the inner EMD's own thresholds are the
        # engine's (and receive these values only if they were passed, as in PyEMD).
        self.range_thr = float(kwargs.get("range_thr", 0.01))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.05))
        self.beta_progress = bool(kwargs.get("beta_progress", True))
        self.noise_backend = _noise_backend(kwargs.get("noise_backend", self._noise_backend_default))
        self.random = _make_rng(self.noise_backend, kwargs.get("seed"))  # PyEMD: np.random.RandomState(seed=...)
        self._rng = self.random  # older name
        self.noise_kind = kwargs.get("noise_kind", "normal")
        self._max_imf = int(kwargs.get("max_imf", 100))  # bound on the stage loop
        self.parallel = parallel                          # no effect: the trials always run as one GPU batch
        self.processes = kwargs.get("processes")          # likewise
        if self.noise_kind not in self.noise_kinds_all:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )
        self.chunk_trials = None if chunk_trials is None else int(chunk_trials)

        if ext_EMD is None:
            self.EMD = EMD(**{k: v for k, v in kwargs.items() if k not in _CEEMDAN_ONLY_KWARGS})
        else:
            self.EMD = as_gpu_emd(ext_EMD)  # like PyEMD, the kwargs are then not used for the EMD
        self._emd = self.EMD  # older name
        self._max_imfs = self.EMD._config.max_imfs  # older name: the initial IMF slots (the buffer grows on demand)

        self.all_noise_EMD = []
        self.C_IMF = None
        self.residue = None
        self._n_err = self._n_runs = 0
        self._first_error = None

    @property
    def max_stages(self):
        """Older name of PyEMD's `max_imf` constructor kwarg: the bound on the stage loop."""
        return self._max_imf

    @max_stages.setter
    def max_stages(self, value):
        self._max_imf = int(value)

    # ------------------------------------------------------------------
    # noise (PyEMD: generate_noise / noise_seed)
    # ------------------------------------------------------------------
    def generate_noise(self, scale, size):
        """Noise with the configured distribution: "normal" with std `scale`, or "uniform" on [-scale/2, scale/2].
        A numpy array with noise_backend="numpy" (PyEMD's numbers), a device array with "cupy"."""
        if self.noise_backend == "numpy":                 # exactly PyEMD's calls
            if self.noise_kind == "normal":
                return self.random.normal(loc=0, scale=scale, size=size)
            if self.noise_kind == "uniform":
                return self.random.uniform(low=-scale / 2, high=scale / 2, size=size)
        else:
            if self.noise_kind == "normal":
                return self.random.normal(0.0, scale, size=size, dtype=cp.float64)
            if self.noise_kind == "uniform":
                return self.random.uniform(low=-scale / 2, high=scale / 2, size=size, dtype=cp.float64)
        raise ValueError(
            "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                str(self.noise_kinds_all)
            )
        )

    def noise_seed(self, seed):
        """Set seed for noise generation."""
        self.random.seed(seed)

    # ------------------------------------------------------------------
    def _auto_chunk(self, N):
        # everything the launcher allocates per signal plus the input row and 3 rows of per-chunk temporaries
        per_trial = self.EMD.bytes_per_signal(N) + 4 * N * 8
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        return max(1, int(0.7 * free) // per_trial)

    def _emd_run(self, batch, T, max_imf=-1, first_trial=0):
        imfs, n_imfs, errors = self.EMD.batch(batch, T, max_imf, trim=False)
        err = cp.asnumpy(errors)
        bad = np.nonzero(err)[0]
        if bad.size:
            self._n_err += int(bad.size)
            if self._first_error is None:
                self._first_error = (int(err[bad[0]]), first_trial + int(bad[0]))
        self._n_runs += batch.shape[0]
        return imfs, n_imfs

    @staticmethod
    def _sum_rows(rows):
        # np.sum(rows, axis=0): 0.0 + row 0 + row 1 + ... in order
        acc = cp.zeros_like(rows[0])
        for r in rows:
            acc += r
        return acc

    def _noise_chunk(self, N, noise, stream, start, stop, keep_on_device):
        # PyEMD generate_noise + _decompose_noise for trials [start, stop): full EMD with T=None, then every row divided
        # by np.std of its first IMF.  Returned as (K, B, N), zeros past a trial's own row count.
        B = stop - start
        if noise is None:
            x = stream.rows(start, stop)
        else:
            x = cp.array(noise[start:stop], dtype=cp.float64)  # used as given, no rescaling
        imfs, n_imfs = self._emd_run(x, None, -1, start)
        del x
        n_h = cp.asnumpy(n_imfs)
        K = int(n_h.max()) if B else 0
        rows = imfs[:, :K, :]
        if self.beta_progress and K:
            # PyEMD: imfs / np.std(imfs[0]) per trial -- numpy's std on the host (row-wise std == per-row np.std)
            std = np.std(cp.asnumpy(imfs[:, 0, :]), axis=1)
            std = np.where(n_h > 0, std, 1.0)
            rows = rows / cp.asarray(std)[:, None, None]
        valid = cp.arange(K)[None, :, None] < cp.asarray(n_h)[:, None, None]
        modes = cp.ascontiguousarray(cp.where(valid, rows, 0.0).transpose(1, 0, 2))   # (K, B, N)
        return (modes if keep_on_device else cp.asnumpy(modes)), cp.asarray(n_h)

    @staticmethod
    def _noisy(base, chunk, k, scale):
        # base + scale * (mode k of each trial); a trial whose noise has no mode k gets base itself (PyEMD skips the
        # addition for it -- adding scale * 0 would turn a -0.0 of base into +0.0).
        modes, n_h = chunk
        B = modes.shape[1]
        if k < modes.shape[0]:
            x = cp.asarray(modes[k]) * scale
            x += base
            lacks = n_h <= k
            if bool(lacks.any()):
                x = cp.where(lacks[:, None], base[None, :], x)
        else:
            x = cp.empty((B, base.shape[0]), dtype=cp.float64)
            x[...] = base
        return cp.ascontiguousarray(x)

    def _end_condition(self, S, cimfs, max_imf):
        if 0 < max_imf <= len(cimfs):
            return True
        R = S - self._sum_rows(cimfs)
        Rh = cp.asnumpy(R)
        # PyEMD ORs these three tests, so the two cheap ones (numpy, on the host -- numpy's own max/min/sum) go first
        # and can skip the EMD run.
        if np.max(Rh) - np.min(Rh) < self.range_thr or np.sum(np.abs(Rh)) < self.total_power_thr:
            return True
        _, n_imfs = self._emd_run(R[None, :], None, max_imf=1)  # PyEMD: emd(R, None, max_imf=1)
        return int(n_imfs[0]) == 1  # one row: too few extrema to sift, or R is already an IMF

    # ------------------------------------------------------------------
    # PyEMD's interface
    # ------------------------------------------------------------------
    def end_condition(self, S, cIMFs, max_imf):
        """PyEMD's end_condition(S, cIMFs, max_imf): True when the CEEMDAN stage loop must stop."""
        S = cp.asarray(S, dtype=cp.float64)
        return self._end_condition(S, [cp.asarray(r, dtype=cp.float64) for r in cIMFs], max_imf)

    def emd(self, S, T=None, max_imf=-1):
        """PyEMD's emd(): one EMD of S with the inner engine."""
        return self.EMD.emd(S, T, max_imf)

    def __call__(self, S, T=None, max_imf=-1, progress=False, noise=None):
        if noise is None and hasattr(max_imf, "shape"):
            # the older positional order was (signal, T, noise, max_imf)
            noise = max_imf
            max_imf = progress if (isinstance(progress, int) and not isinstance(progress, bool)) else -1
            progress = False
        S_host = np.asarray(cp.asnumpy(S) if isinstance(S, cp.ndarray) else S)
        if S_host.ndim != 1 or S_host.dtype != np.float64:
            raise ValueError(f"expected a 1-D float64 signal, got shape {S_host.shape}, dtype {S_host.dtype}")
        N = S_host.shape[0]
        if noise is not None:
            if not hasattr(noise, "shape"):
                noise = np.asarray(noise, dtype=np.float64)
            if tuple(noise.shape) != (self.trials, N):
                raise ValueError(f"noise must have shape (trials, N) = ({self.trials}, {N}), got {noise.shape}")

        self._n_err = self._n_runs = 0
        self._first_error = None
        chunk = min(self.chunk_trials or self._auto_chunk(N), self.trials)

        scale_s = float(np.std(S_host))                     # numpy's std (pairwise sums), not the GPU's
        S = cp.asarray(S_host) / scale_s                    # PyEMD: S = S / scale_s

        stream = None
        if noise is None:
            stream = _K.NoiseStream(lambda rows: self.generate_noise(self.noise_scale, (rows, N)), N,
                                    exact=(self.noise_backend == "numpy"))
        # the decomposed noise stays on the GPU when it fits comfortably, else on the host
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        est = 8 * N * self.trials * (int(np.ceil(np.log2(max(N, 2)))) + 4)
        on_device = est < 0.3 * free
        # T goes to the first-IMF and stage EMDs; noise and end-condition EMDs use T=None, as PyEMD does.
        modes = [(s, self._noise_chunk(N, noise, stream, s, min(s + chunk, self.trials), on_device))
                 for s in range(0, self.trials, chunk)]

        # cIMF 1: PyEMD _eemd(S, T, max_imf=1): E_IMF = 0; E_IMF += EMD(S + epsilon*mode_0) per trial; / trials
        total = cp.zeros(N, dtype=cp.float64)
        for s, m in modes:
            x = self._noisy(S, m, 0, self.epsilon)
            imfs, n_imfs = self._emd_run(x, T, 1, s)
            del x
            sel = cp.where(n_imfs > 0, 0, -1).astype(cp.int32)
            _K.accumulate_rows(total, imfs, sel)
        cimfs = [total / self.trials]
        prev_res = S - cimfs[0]
        stages = range(self._max_imf)
        if progress and tqdm is not None:
            stages = tqdm(stages, desc="cIMF", total=self._max_imf)
        for _ in stages:
            if self._end_condition(S, cimfs, max_imf):
                break
            beta = self.epsilon * float(np.std(cp.asnumpy(prev_res)))
            local_mean = cp.zeros(N, dtype=cp.float64)
            k = len(cimfs)
            for s, m in modes:
                x = self._noisy(prev_res, m, k, beta)
                imfs, n_imfs = self._emd_run(x, T, 1, s)
                del x
                # PyEMD: local_mean += emd(res, T, max_imf=1)[-1] / trials -- the LAST row, divided, summed in order
                sel = cp.where(n_imfs >= 1, n_imfs - 1, -1).astype(cp.int32)
                _K.accumulate_rows(local_mean, imfs, sel, div=float(self.trials))
            cimfs.append(prev_res - local_mean)
            prev_res = local_mean

        components = cp.stack(cimfs + [S - self._sum_rows(cimfs)]) * scale_s
        self._S_norm = S
        self._scale_s = scale_s
        return components, self._n_err

    def ceemdan(self, S, T=None, max_imf=-1, progress=False, noise=None):
        """PyEMD's ceemdan(): the complete ensemble IMFs as a numpy array of shape (cIMFs + 1, N), the last row
        being S minus the sum of the cIMFs. Also sets C_IMF and residue.  Raises where PyEMD raises."""
        S_host = np.asarray(cp.asnumpy(S) if isinstance(S, cp.ndarray) else S, dtype=np.float64)
        components, n_err = GPUBatchCEEMDAN.__call__(self, S_host, T, max_imf, progress, noise)
        if n_err:
            code, trial = self._first_error
            raise_for_error(code, f"trial {trial}")
        self.C_IMF = cp.asnumpy(components)
        # PyEMD: self.residue = S * scale_s - np.sum(self.C_IMF, axis=0), with S the normalised signal
        self.residue = cp.asnumpy(self._S_norm) * self._scale_s - np.sum(self.C_IMF, axis=0)
        return self.C_IMF

    def get_imfs_and_residue(self):
        """Provides access to the cIMFs and the residue of the last ceemdan() call."""
        if self.C_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.C_IMF, self.residue


class CEEMDAN(GPUBatchCEEMDAN):
    """Drop-in replacement for PyEMD.CEEMDAN: __call__ is ceemdan(); the noise is numpy's RandomState stream, as in
    PyEMD (noise_backend="cupy" draws it on the GPU instead: faster, but not PyEMD's numbers)."""

    _noise_backend_default = "numpy"

    def __call__(self, S, T=None, max_imf=-1, progress=False):
        return self.ceemdan(S, T=T, max_imf=max_imf, progress=progress)

"""eemd.py -- EEMD on top of GPUBatchEMD, with PyEMD's EEMD interface.

TWO CLASSES
    EEMD           drop-in replacement for PyEMD.EEMD: EEMD(...)(S) / eemd(S) return the ensemble IMFs as a numpy
                   array; all_imfs / ensemble_std() are available; the noise is numpy's RandomState stream, as in
                   PyEMD, so with the same seed the result is PyEMD's (parallel=False) bit for bit.
    GPUBatchEEMD   the same engine tuned for throughput: noise drawn on the GPU (noise_backend="cupy"), trials' IMFs
                   not kept unless asked (store_all_imfs=True); __call__ returns device arrays
                   (mean_imfs, counts, n_errors).

CONSTRUCTOR -- the same as PyEMD's EEMD:

    EEMD(trials=100, noise_width=0.05, ext_EMD=None, parallel=True, **kwargs)

    trials         number of noisy EMD runs
    noise_width    noise std relative to the signal's peak-to-peak amplitude
    ext_EMD        an EMD engine to use instead of building one from the kwargs: an EMD / GPUBatchEMD, or a CPU
                   PyEMD.EMD (its settings are copied onto a GPU engine)
    parallel       accepted for compatibility; the trials always run as one GPU batch
    kwargs         separate_trends=False  noise_kind="normal" ("normal" or "uniform")  processes=None (no effect
                   on a GPU), and everything else is forwarded to the EMD engine exactly as PyEMD forwards it to
                   EMD(**kwargs): spline_kind, nbsym, the thresholds, FIXE, FIXE_H, MAX_ITERATION, ...
                   GPU-only: seed (the noise generator's seed), noise_backend ("numpy": PyEMD's stream, drawn on the
                   CPU; "cupy": drawn on the GPU, faster, different numbers), chunk_trials (trials per GPU batch;
                   default: as many as fit in memory), store_all_imfs (keep every trial's IMFs: all_imfs,
                   ensemble_std).

EXACTNESS
    Given the same noise (pass it with eemd(S, noise=...): shape (trials, N), absolute scale), the result is PyEMD's
    bit for bit: every trial's EMD is (see kernels.py), and the ensemble mean is formed in PyEMD's order -- per IMF
    order, the trials that have it, summed one after another in trial order from 0.0 (numpy's axis-0 reduction), then
    divided by their count.  The noise itself comes from CuPy's generator (PyEMD's numpy RandomState stream cannot
    be reproduced on the GPU at speed); its distribution is PyEMD's: normal with std noise_width * (max(S) - min(S)),
    or uniform on [-scale/2, scale/2].

THE BATCH CALL (GPUBatchEEMD)
    mean_imfs, counts, n_errors = GPUBatchEEMD(...)(signal)
    n_errors counts trials PyEMD itself could not decompose (it would raise); 0 in practice.  eemd() raises instead.
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


class _Accumulator:
    """Running per-order sums of the trials' IMF rows (and of the trends, with separate_trends), in trial order."""

    def __init__(self, N, separate_trends, store_all):
        self.N = N
        self.separate = separate_trends
        self.sums = []    # one (N,) device array per order, grown on demand
        self.counts = []  # per-order number of trials that have that order
        self.trend_sum = None
        self.trend_count = 0
        self.all = {} if store_all else None  # order -> list of per-trial rows (host), for all_imfs
        self.all_trend = [] if store_all else None
        self.first_error = 0
        self.n_errors = 0

    def add(self, imfs, n_imfs):
        """Add one batch (trials in order): imfs [B, slots, N], n_imfs [B] (device)."""
        n_h = cp.asnumpy(n_imfs)
        B = n_h.shape[0]
        if self.separate:
            # PyEMD: imfs, trend = EMD.get_imfs_and_trend(); the trend is the LAST row whether it is the residue
            # or (when the residue is ~0) the last IMF, and the IMFs are every row before it.
            n_rows_h = np.maximum(n_h - 1, 0)
            sel = np.where(n_h >= 1, n_h - 1, -1).astype(np.int32)
            if (sel >= 0).any():
                if self.trend_sum is None:
                    self.trend_sum = cp.zeros(self.N, dtype=cp.float64)
                _K.accumulate_rows(self.trend_sum, imfs, cp.asarray(sel))
                self.trend_count += int((sel >= 0).sum())
        else:
            n_rows_h = n_h
        top = int(n_rows_h.max()) if B else 0
        for k in range(top):
            if k == len(self.sums):  # a trial produced more orders than any so far
                self.sums.append(cp.zeros(self.N, dtype=cp.float64))
                self.counts.append(0)
            sel = np.where(n_rows_h > k, k, -1).astype(np.int32)
            _K.accumulate_rows(self.sums[k], imfs, cp.asarray(sel))
            self.counts[k] += int((n_rows_h > k).sum())

        if self.all is not None:
            host = cp.asnumpy(imfs)
            for b in range(B):
                for k in range(int(n_rows_h[b])):
                    self.all.setdefault(k, []).append(host[b, k].copy())
                if self.separate and n_h[b] >= 1:
                    self.all_trend.append(host[b, int(n_h[b]) - 1].copy())

    def result(self):
        """(mean rows [orders(+1), N] device array, per-order trial counts as a list)."""
        counts = list(self.counts)
        rows = [s / float(c) for s, c in zip(self.sums, counts)]   # np.mean: sum / count
        if self.separate and self.trend_count:
            rows.append(self.trend_sum / float(self.trend_count))
            counts.append(self.trend_count)
        mean = cp.stack(rows) if rows else cp.zeros((0, self.N), dtype=cp.float64)
        return mean, counts


class GPUBatchEEMD:
    noise_kinds_all = ["normal", "uniform"]
    _store_all_default = False
    _noise_backend_default = "cupy"

    def __init__(self, trials=100, noise_width=0.05, ext_EMD=None, parallel=True, **kwargs):
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        chunk_trials = kwargs.pop("chunk_trials", None)
        if chunk_trials is not None and chunk_trials < 1:
            raise ValueError(f"chunk_trials must be >= 1 or None, got {chunk_trials}")
        seed = kwargs.pop("seed", None)
        self.noise_backend = _noise_backend(kwargs.pop("noise_backend", self._noise_backend_default))
        self.store_all_imfs = bool(kwargs.pop("store_all_imfs", self._store_all_default))

        # ---- PyEMD.EEMD.__init__ ----
        self.trials = int(trials)
        self.noise_width = float(noise_width)
        self.separate_trends = bool(kwargs.pop("separate_trends", False))
        self.noise_kind = kwargs.pop("noise_kind", "normal")
        self.parallel = parallel                  # no effect: the trials always run as one GPU batch
        self.processes = kwargs.pop("processes", None)  # likewise
        if self.noise_kind not in self.noise_kinds_all:
            raise ValueError(
                "Unsupported noise kind. Please assigned `noise_kind` to be one of these: {0}".format(
                    str(self.noise_kinds_all)
                )
            )
        self.chunk_trials = None if chunk_trials is None else int(chunk_trials)
        self.random = _make_rng(self.noise_backend, seed)  # PyEMD: np.random.RandomState()
        self._rng = self.random  # older name

        if ext_EMD is None:
            self.EMD = EMD(**kwargs)
        else:
            self.EMD = as_gpu_emd(ext_EMD)  # like PyEMD, the kwargs are then not used for the EMD
        self._emd = self.EMD  # older name
        self._max_imfs = self.EMD._config.max_imfs  # older name: the initial IMF slots (the buffer grows on demand)

        self.E_IMF = None
        self.residue = None
        self._counts = None
        self._all_imfs = None
        self.last_errors = 0

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
        # Per trial: everything the launcher allocates for one signal (IMF output rows, working scratch), plus the
        # input row and a noise row.  Budget 70% of what's free (plus CuPy's cached pool).
        per_trial = self.EMD.bytes_per_signal(N) + 2 * N * 8
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        return max(1, int(0.7 * free) // per_trial)

    def _run_chunk(self, signal, T, noise, stream, start, stop, max_imf, acc):
        """Run trials [start, stop) as one batch and add their rows into `acc` (in trial order)."""
        if noise is None:
            trial_in = stream.rows(start, stop)
        else:
            trial_in = cp.array(noise[start:stop], dtype=cp.float64)
        trial_in += signal                         # PyEMD: S + noise (addition commutes exactly)
        imfs, n_imfs, errors = self.EMD.batch(trial_in, T, max_imf, trim=False)
        del trial_in
        acc.add(imfs, n_imfs)
        err = cp.asnumpy(errors)
        bad = np.nonzero(err)[0]
        if bad.size:
            acc.n_errors += int(bad.size)
            if not acc.first_error:
                acc.first_error = (int(err[bad[0]]), start + int(bad[0]))

    def __call__(self, S, T=None, max_imf=-1, progress=False, noise=None):
        if noise is None and hasattr(max_imf, "shape"):  # the older positional order was (signal, T, noise)
            noise, max_imf = max_imf, -1
        signal = cp.asarray(S)
        if signal.ndim != 1 or cp.dtype(signal.dtype) != cp.dtype(cp.float64):
            raise ValueError(f"expected a 1-D float64 signal, got shape {signal.shape}, dtype {signal.dtype}")
        N = signal.shape[0]

        stream = None
        if noise is not None:
            if not hasattr(noise, "shape"):
                noise = np.asarray(noise, dtype=np.float64)
            if tuple(noise.shape) != (self.trials, N):
                raise ValueError(f"noise must have shape (trials, N) = ({self.trials}, {N}), got {noise.shape}")
        else:
            scale = self.noise_width * abs(float(signal.max() - signal.min()))  # PyEMD's noise scale
            stream = _K.NoiseStream(lambda rows: self.generate_noise(scale, (rows, N)), N,
                                    exact=(self.noise_backend == "numpy"))

        acc = _Accumulator(N, self.separate_trends, self.store_all_imfs)
        chunk = min(self.chunk_trials or self._auto_chunk(N), self.trials)
        bar = tqdm(total=self.trials, desc="EEMD") if (progress and tqdm is not None) else None
        for start in range(0, self.trials, chunk):
            stop = min(start + chunk, self.trials)
            self._run_chunk(signal, T, noise, stream, start, stop, max_imf, acc)
            if bar is not None:
                bar.update(stop - start)
        if bar is not None:
            bar.close()

        mean_imfs, counts = acc.result()
        self._counts = np.asarray(counts, dtype=np.int64)
        self._all_imfs = self._stored_all_imfs(acc)
        self.last_errors = acc.n_errors
        self._first_error = acc.first_error
        return mean_imfs, cp.asarray(self._counts, dtype=cp.int32), acc.n_errors

    @staticmethod
    def _stored_all_imfs(acc):
        if acc.all is None:
            return None
        out = {k: np.array(rows) for k, rows in acc.all.items()}
        if acc.separate and acc.all_trend:
            out[len(out)] = np.array(acc.all_trend)  # PyEMD: the trend becomes the last order
        return out

    # ------------------------------------------------------------------
    # PyEMD's interface
    # ------------------------------------------------------------------
    def emd(self, S, T=None, max_imf=-1):
        """PyEMD's EEMD.emd(): one plain EMD of S with the ensemble's EMD engine (no noise)."""
        return self.EMD.emd(S, T, max_imf)

    def eemd(self, S, T=None, max_imf=-1, progress=False, noise=None):
        """PyEMD's eemd(): the ensemble IMFs as a numpy array of shape (orders, N) (with separate_trends the mean
        trend is the last row). Also sets E_IMF and residue (= S - sum of the E_IMF rows).  Raises where PyEMD
        raises (a trial PyEMD cannot decompose)."""
        S_host = np.asarray(cp.asnumpy(S) if isinstance(S, cp.ndarray) else S, dtype=np.float64)
        mean_imfs, _, n_err = GPUBatchEEMD.__call__(self, S_host, T, max_imf, progress, noise)
        if n_err:
            code, trial = self._first_error
            raise_for_error(code, f"trial {trial}")
        self.E_IMF = cp.asnumpy(mean_imfs)
        self.residue = S_host - np.sum(self.E_IMF, axis=0)
        return self.E_IMF

    def get_imfs_and_residue(self):
        """Provides access to the ensemble IMFs and the residue of the last eemd() call."""
        if self.E_IMF is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.E_IMF, self.residue

    @property
    def all_imfs(self):
        """A dictionary with all computed imfs per given order (needs store_all_imfs=True; EEMD keeps them)."""
        if self._all_imfs is None:
            if self._counts is None:
                return {}
            raise RuntimeError("all_imfs needs the trials' IMFs; construct with store_all_imfs=True")
        return self._all_imfs

    def ensemble_count(self):
        """Count of imfs observed for given order, e.g. 1st proto-imf, in the whole ensemble."""
        if self._counts is None:
            return []
        return [int(c) for c in self._counts]

    def ensemble_mean(self):
        """Pointwise mean over computed ensemble. Same as the output of eemd()."""
        if self.E_IMF is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.E_IMF

    def ensemble_std(self):
        """Pointwise standard deviation over computed ensemble (needs the trials' IMFs, see all_imfs)."""
        return np.array([imfs.std(axis=0) for imfs in self.all_imfs.values()])


class EEMD(GPUBatchEEMD):
    """Drop-in replacement for PyEMD.EEMD: __call__ is eemd(); the noise is numpy's RandomState stream (PyEMD's
    numbers for the same seed); every trial's IMFs are kept (all_imfs, ensemble_std()) as PyEMD keeps them.
    For maximum speed pass noise_backend="cupy" and store_all_imfs=False (or use GPUBatchEEMD)."""

    _store_all_default = True
    _noise_backend_default = "numpy"

    def __call__(self, S, T=None, max_imf=-1, progress=False):
        return self.eemd(S, T=T, max_imf=max_imf, progress=progress)

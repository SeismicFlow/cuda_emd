"""
emd.py -- GPU batch EMD with PyEMD's interface (thin host-side wrapper around the kernel in kernels.py).

TWO CLASSES
    EMD            the drop-in replacement for PyEMD.EMD: same constructor, same methods, same return values.
                   `from PyEMD import EMD`  ->  `from cuda_emd import EMD`.
    GPUBatchEMD    the same engine with the batch interface: __call__ takes a [batch, N] array and returns device
                   arrays (no host round trip), for pipelines that stay on the GPU.  EMD is a GPUBatchEMD.

CONSTRUCTOR -- the same as PyEMD's EMD:

    EMD(spline_kind="cubic", nbsym=2, **kwargs)

    spline_kind        "cubic" (default), "linear", "slinear", "quadratic", "cubic_hermite", "pchip", "akima"
    nbsym              boundary mirroring depth (the kwarg `nbsym` overrides the positional one, as in PyEMD)
    kwargs, all with PyEMD's defaults and all plain attributes you may change between calls:
      energy_ratio_thr=0.2  std_thr=0.2  svar_thr=0.001           the IMF convergence tests (check_imf)
      total_power_thr=0.005 range_thr=0.001                       the end condition of the whole EMD
      FIXE=0  FIXE_H=0  MAX_ITERATION=1000                        the sifting stop rules
      extrema_detection="simple"                                  "simple" or "parabol"
      scale_factor=1.0  DTYPE=np.float64                          accepted for compatibility (see below)
    GPU-only kwargs: blocks_per_signal (None = automatic), max_imfs (initial IMF slots), threads_per_block, config.

    Unknown kwargs are ignored, as PyEMD ignores them, but with a warning, so a typo does not go unnoticed.

METHODS (PyEMD's)
    emd(S, T=None, max_imf=-1)     one 1-D signal in, an (IMFs [+ residue], N) numpy array out; sets .imfs, .residue
    __call__(S, T=None, max_imf=-1)  EMD: the same as emd() (a 2-D S gives one such array per row, as a list);
                                   GPUBatchEMD: the batch call, see below
    get_imfs_and_residue(), get_imfs_and_trend()
    find_extrema, prepare_points, spline_points, extract_max_min_spline, check_imf, end_condition, ... -- PyEMD's
                                   internal building blocks are fused into the GPU sift loop; on EMD they are
                                   still available (computed by PyEMD on the CPU) when PyEMD is installed.

THE RESULT IS PyEMD'S, BIT FOR BIT
    The kernel reproduces every floating-point operation of PyEMD and every quantity a decision is taken on (numpy's
    pairwise sums, Python's max/min, numpy's NaN-propagating max/min), so the IMFs, their number and the residue are
    the ones PyEMD computes with the installed numpy / scipy -- see kernels.py, EXACTNESS.  Two things differ on purpose:
    * float64 only: PyEMD computes in the input's dtype; here float32 input is promoted to float64 (emd()) or
      rejected (batch call).
    * Where PyEMD raises for a signal (more than two extrema but all of one kind; knots scipy refuses), emd() raises the
      same kind of exception; the batch call cannot raise for one signal of many and reports it in `errors` instead.
    A sift that reaches MAX_ITERATION is NOT an error: PyEMD accepts the IMF as it stands and moves on, and so does
    the kernel -- same result, bit for bit.  It is logged at INFO level, as PyEMD logs it.

BATCH CALL (GPUBatchEMD)

    imfs, n_imfs, errors = GPUBatchEMD()(signals)        # signals: [batch, N] float64, numpy or cupy

    imfs:    cupy float64 [batch, slots, N]; signal i's rows are imfs[i, :n_imfs[i]] -- PyEMD's emd() output
             (IMFs, then the residue as the last row unless it is all-close to zero).  `slots` is sized automatically:
             a signal that needs more rows than the buffer has is re-run with a bigger one, never truncated.
    n_imfs:  cupy int32 [batch]
    errors:  cupy int32 [batch], 0 for every signal PyEMD decomposes; nonzero only where PyEMD itself raises an
             exception (kernels.ERR_NOEXT / ERR_SPLINE).  Nothing else is ever reported.
    last_debug = (iterations, extNo, nzm) per IMF (cupy int32 [batch, slots], -1 where unused), last_imf_counts (the
    number of real IMFs per signal) and last_max_iteration_hits (IMFs that reached MAX_ITERATION) are kept for
    diagnostics.

Every signal in the batch shares the same T (time) axis.

COMPILATION AND WARMUP: the kernel is compiled (NVRTC, then the driver's PTX JIT) the first time it is launched with a
given (nbsym, spline_kind, extrema_detection) -- a one-time cost of a few seconds, cached on disk by CuPy afterwards.
Call `warmup()` once before timing anything.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import replace

import cupy as cp
import numpy as np

try:
    from . import kernels as _K
    from .kernels import EMDKernelConfig  # package layout (cuda_emd/ as a subpackage)
except ImportError:
    import kernels as _K
    from kernels import EMDKernelConfig  # flat layout: emd.py and kernels.py side by side

# Every keyword GPUBatchEMD understands: PyEMD's EMD kwargs, then the GPU-only ones. EEMD / CEEMDAN forward the
# kwargs they do not use themselves to GPUBatchEMD, as PyEMD forwards them to EMD(**kwargs).
PYEMD_EMD_KWARGS = (
    "energy_ratio_thr", "std_thr", "svar_thr", "total_power_thr", "range_thr", "nbsym", "scale_factor",
    "extrema_detection", "DTYPE", "FIXE", "FIXE_H", "MAX_ITERATION", "spline_kind",
)
GPU_ONLY_KWARGS = ("config", "auto_imfs", "max_extrema", "max_imfs", "threads_per_block", "max_iteration",
                   "blocks_per_signal")
EMD_KWARGS = PYEMD_EMD_KWARGS + GPU_ONLY_KWARGS

ERR_NOEXT, ERR_SPLINE, ERR_CAP, ERR_NONTERM = _K.ERR_NOEXT, _K.ERR_SPLINE, _K.ERR_CAP, _K.ERR_NONTERM

logger = logging.getLogger(__name__)


class EMDOverflowWarning(UserWarning):
    """Kept so that code importing it keeps working.  The library no longer issues it: a sift that reaches
    MAX_ITERATION is PyEMD's own (logged) behaviour, the IMF buffer always grows, and signals PyEMD cannot
    decompose are reported through `errors` (batch) or an exception (emd())."""


def raise_for_error(code: int, where: str = "") -> None:
    """Raise the exception PyEMD raises for a signal with this kernel error code (no-op for 0)."""
    code = int(code)
    if not code:
        return
    w = f" ({where})" if where else ""
    if code & ERR_NOEXT:
        raise IndexError("index 0 is out of bounds for axis 0 with size 0" + w +
                         " -- PyEMD raises this too: the signal has more than two extrema, but all of one kind")
    if code & ERR_SPLINE:
        raise ValueError("the envelope spline cannot be built" + w + " -- PyEMD/scipy raise here too (too few knots, "
                         "knots that are not strictly increasing, or an envelope that does not cover the time axis)")
    if code & ERR_NONTERM:
        raise RuntimeError("the decomposition did not end within N + 2 IMFs" + w + " (PyEMD would not end either)")
    raise RuntimeError(f"internal error code {code}{w}")


class GPUBatchEMD:
    def __init__(self, spline_kind: str = "cubic", nbsym: int = 2, **kwargs):
        # ---- GPU-only keywords, and the legacy GPUBatchEMD(config) call ----
        config = kwargs.pop("config", None)
        if isinstance(spline_kind, EMDKernelConfig):
            if config is not None:
                raise TypeError("pass the EMDKernelConfig either positionally or as config=..., not both")
            config, spline_kind = spline_kind, "cubic"
        kwargs.pop("auto_imfs", None)            # accepted for compatibility: the IMF buffer always grows on demand
        self.blocks_per_signal = kwargs.pop("blocks_per_signal", None)
        gpu_fields = {k: kwargs.pop(k) for k in ("max_extrema", "max_imfs", "threads_per_block") if k in kwargs}

        unknown = sorted(set(kwargs) - set(EMD_KWARGS))
        if unknown:
            warnings.warn(
                f"{type(self).__name__} ignores the unknown keyword argument(s) {unknown} (PyEMD's EMD ignores them "
                f"too, silently); the ones it understands are {sorted(PYEMD_EMD_KWARGS)}.",
                stacklevel=2,
            )

        def deferred(value, pyemd_default, config_value):
            # A value equal to PyEMD's default is indistinguishable from "not given": if a config was supplied
            # it decides then, otherwise the explicit value wins.
            return config_value if (config is not None and value == pyemd_default) else value

        # ---- PyEMD.EMD.__init__, name for name ----
        self.energy_ratio_thr = float(kwargs.get("energy_ratio_thr", 0.2))
        self.std_thr = float(kwargs.get("std_thr", 0.2))
        self.svar_thr = float(kwargs.get("svar_thr", 0.001))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.005))
        self.range_thr = float(kwargs.get("range_thr", 0.001))
        nbsym = int(kwargs.get("nbsym", nbsym))
        self.scale_factor = float(kwargs.get("scale_factor", 1.0))
        extrema_detection = kwargs.get("extrema_detection", "simple")
        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.FIXE = int(kwargs.get("FIXE", 0))
        self.FIXE_H = int(kwargs.get("FIXE_H", 0))
        max_iteration = int(kwargs.get("MAX_ITERATION", kwargs.get("max_iteration", 1000)))
        if config is not None:
            nbsym = deferred(nbsym, 2, config.nbsym)
            spline_kind = deferred(spline_kind, "cubic", config.spline_kind)
            extrema_detection = deferred(extrema_detection, "simple", config.extrema_detection)
            max_iteration = deferred(max_iteration, 1000, config.max_iteration)
        self.nbsym = nbsym
        self.spline_kind = spline_kind
        self.extrema_detection = extrema_detection
        self.MAX_ITERATION = max_iteration

        # results of the last emd() call, as in PyEMD
        self.imfs = None
        self.residue = None

        if np.dtype(self.DTYPE) != np.dtype(np.float64):
            raise NotImplementedError(
                f"GPUBatchEMD is float64 only, got DTYPE={self.DTYPE!r} (PyEMD overwrites DTYPE with the input "
                "signal's dtype on every call; pass float64 signals)."
            )

        # ---- GPU configuration ----
        if config is None:
            config = EMDKernelConfig(**gpu_fields)
        elif gpu_fields:
            config = replace(config, **gpu_fields)
        self._config = config  # the base config; the effective one also follows nbsym / spline_kind / extrema_detection
        self._launchers = {}   # compile-time settings -> built launcher
        self.last_debug = None
        self.last_imf_counts = None
        self.last_max_iteration_hits = None
        self.last_errors = None

        self._effective_config().validate()  # reject a bad spline_kind / extrema_detection now, not at the first call

    # ------------------------------------------------------------------
    # configuration plumbing
    # ------------------------------------------------------------------
    def _effective_config(self) -> EMDKernelConfig:
        """The compile-time configuration for the CURRENT attribute values (nbsym, spline_kind and
        extrema_detection are plain, changeable attributes, as in PyEMD)."""
        return replace(
            self._config,
            nbsym=int(self.nbsym),
            spline_kind=str(self.spline_kind).lower(),
            extrema_detection=self.extrema_detection,
        )

    def _get_launcher(self):
        cfg = self._effective_config()
        key = (cfg.nbsym, cfg.threads_per_block, cfg.spline_kind, cfg.extrema_detection)
        launcher = self._launchers.get(key)
        if launcher is None:
            launcher = cfg.build()  # built lazily, compiled at first launch -- see warmup()
            self._launchers[key] = launcher
        return launcher

    def warmup(self, N: int = 64) -> None:
        """Force compilation now, via one tiny real call, so it doesn't land inside whatever the next call is."""
        dummy = cp.zeros((1, N), dtype=cp.float64)
        GPUBatchEMD.__call__(self, dummy)

    def _imf_slots(self, N: int) -> int:
        """Initial IMF output slots for an N-sample signal: enough for a typical EMD (about log2(N) IMFs plus the
        residue) with margin; the buffer grows for the rare signal that needs more."""
        return max(int(self._config.max_imfs), int(math.ceil(math.log2(max(int(N), 2)))) + 3)

    def bytes_per_signal(self, N: int) -> int:
        """Device memory one N-sample signal needs inside a call (IMF output + working scratch), so a caller with a
        big batch (EEMD / CEEMDAN) can size its chunks against free memory."""
        return self._effective_config().bytes_per_signal(int(N), self._imf_slots(N))

    def _timeline(self, N: int, T) -> "cp.ndarray":
        """PyEMD.emd(): `if T is None or extrema_detection == "simple": T = get_timeline(len(S))`, then
        `T = self._normalize_time(T)` = (T - T[0]) / min(diff(T))."""
        if T is not None:
            T = cp.asarray(T, dtype=cp.float64)
            if T.shape != (N,):
                raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(N, T.size))
        if T is None or self.extrema_detection == "simple":
            return cp.arange(N, dtype=cp.float64)
        d = cp.diff(T)
        if not bool(cp.all(d != 0)):
            # PyEMD: `assert np.all(d != 0), "All time domain values needs to be unique"` -- same exception type
            raise AssertionError("All time domain values needs to be unique")
        return (T - T[0]) / cp.min(d)

    # ------------------------------------------------------------------
    # the batch call
    # ------------------------------------------------------------------
    def batch(self, S, T=None, max_imf: int = -1, trim: bool = True):
        """Decompose a batch: S is [batch, N] (or one 1-D signal). Returns (imfs, n_imfs, errors) as described in the
        module docstring. `max_imf` is PyEMD's: stop after that many IMFs (<= 0: no limit).  trim=True returns only as
        many row slots as the longest decomposition uses (less to copy to the host)."""
        signals = cp.asarray(S)
        if signals.ndim == 1:
            signals = signals[None, :]
        if cp.dtype(signals.dtype) != cp.dtype(cp.float64):
            raise ValueError(
                f"GPUBatchEMD is float64 only, got {signals.dtype}. Cast explicitly first, e.g. "
                f"signals.astype(cp.float64)."
            )
        launcher = self._get_launcher()
        batch, N = signals.shape
        T_dev = self._timeline(N, T)
        slots = self._imf_slots(N)

        imfs, n_imfs, errors, debug_n, debug_extno, debug_nzm = launcher.run(
            cp.ascontiguousarray(signals), T_dev, max_imfs=slots,
            std_thr=self.std_thr, svar_thr=self.svar_thr, energy_ratio_thr=self.energy_ratio_thr,
            total_power_thr=self.total_power_thr, range_thr=self.range_thr,
            max_iteration=self.MAX_ITERATION, FIXE=self.FIXE, FIXE_H=self.FIXE_H, max_imf=max_imf,
            blocks_per_signal=self.blocks_per_signal,
        )
        self.last_debug = (debug_n, debug_extno, debug_nzm)
        self.last_imf_counts = launcher.last_imf_counts
        self.last_max_iteration_hits = launcher.last_maxit
        self.last_errors = errors
        if trim:
            rows = max(1, int(n_imfs.max())) if batch else 1
            if rows < imfs.shape[1]:
                imfs = cp.ascontiguousarray(imfs[:, :rows])
        if logger.isEnabledFor(logging.INFO):
            hits = int(cp.sum(launcher.last_maxit))
            if hits:
                logger.info("Max iterations reached for %d IMF(s) of %d signal(s). Continuing with another IMF "
                            "(PyEMD's behaviour).", hits, batch)
        return imfs, n_imfs, errors

    def __call__(self, S, T=None, max_imf: int = -1):
        return self.batch(S, T, max_imf)

    # ------------------------------------------------------------------
    # self-test: are the envelopes bit-for-bit what THIS machine's scipy computes?
    # ------------------------------------------------------------------
    def selftest(self, verbose: bool = False, warn: bool = True) -> dict:
        """Check, on this machine and for the CURRENT settings (spline_kind, nbsym, extrema_detection), that the
        GPU envelopes are bit for bit what the installed numpy / scipy compute.  It builds the first envelope pair of
        a few fixed signals on the GPU (extrema + mirroring + spline, `run_envelope_only`) and compares it, with
        `np.array_equal`, with PyEMD's own spline_points arithmetic evaluated by scipy on the same knots.

        Returns a dict:  ok (bool), checked (envelopes compared), mismatches (list of strings), note (str)."""
        report = {"ok": True, "checked": 0, "mismatches": [], "note": ""}
        try:
            from scipy.interpolate import (Akima1DInterpolator, CubicHermiteSpline, CubicSpline,
                                           PchipInterpolator, interp1d)
        except Exception as ex:  # pragma: no cover
            report["note"] = f"skipped: scipy is not importable ({ex})"
            return report

        kind = str(self.spline_kind).lower()
        mode = self.extrema_detection

        def cubic_spline_3pts(x, y, T):  # PyEMD.splines.cubic_spline_3pts, verbatim arithmetic
            x0, x1, x2 = x
            y0, y1, y2 = y
            x1x0, x2x1 = x1 - x0, x2 - x1
            y1y0, y2y1 = y1 - y0, y2 - y1
            _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1
            m11, m12, m13 = 2 * _x1x0, _x1x0, 0
            m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
            m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1
            v1 = 3 * y1y0 * _x1x0 * _x1x0
            v3 = 3 * y2y1 * _x2x1 * _x2x1
            v2 = v1 + v3
            M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
            v = np.array([v1, v2, v3]).T
            k = np.linalg.solve(M, v)
            a1 = k[0] * x1x0 - y1y0
            b1 = -k[1] * x1x0 + y1y0
            a2 = k[1] * x2x1 - y2y1
            b2 = -k[2] * x2x1 + y2y1
            t1 = (T[np.r_[T >= x0] & np.r_[T < x1]] - x0) / x1x0
            t2 = (T[np.r_[T >= x1] & np.r_[T <= x2]] - x1) / x2x1
            t11, t22 = 1.0 - t1, 1.0 - t2
            q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
            q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
            return np.append(q1, q2)

        def reference(T, pos, val):  # PyEMD.EMD.spline_points, same scipy calls
            t = T[(T >= pos[0]) & (T <= pos[-1])]
            if kind == "akima":
                return Akima1DInterpolator(pos, val)(t)
            if kind == "cubic":
                return CubicSpline(pos, val)(t) if len(pos) > 3 else cubic_spline_3pts(pos, val, t)
            if kind == "pchip":
                return PchipInterpolator(pos, val)(t)
            if kind == "cubic_hermite":
                return CubicHermiteSpline(pos, val, np.gradient(val, pos))(t)
            return interp1d(pos, val, kind=kind)(t).astype(np.float64)

        N = 160
        rng = np.random.RandomState(20240607)
        t = np.arange(N, dtype=np.float64)
        sigs = np.stack([
            rng.randn(N),
            np.sin(2 * np.pi * t / 37) + 0.5 * np.sin(2 * np.pi * t / 11.3) + 0.3 * rng.randn(N),
            (1 + 0.5 * np.sin(2 * np.pi * t / N * 3)) * np.sin(2 * np.pi * (0.01 * t + 3e-4 * t * t)) + 0.05 * rng.randn(N),
            np.cumsum(rng.randn(N)),
            1e3 * rng.randn(N) + 5.0,
            -np.sin(2 * np.pi * 1.5 * t / N + 0.3),   # one maximum: a three-knot envelope (cubic's special case)
        ])
        T_user = np.linspace(0.0, 2.0 * np.pi, N) if mode == "parabol" else None
        try:
            T_dev = self._timeline(N, T_user)
            res = self._get_launcher().run_envelope_only(cp.asarray(sigs), T_dev)
            env = (cp.asnumpy(res[0]), cp.asnumpy(res[1]))
            degenerate = cp.asnumpy(res[3])
            kpos = (cp.asnumpy(res[4]), cp.asnumpy(res[6]))
            kval = (cp.asnumpy(res[5]), cp.asnumpy(res[7]))
            nk = cp.asnumpy(res[8])
            Tn = cp.asnumpy(T_dev)
        except Exception as ex:  # pragma: no cover
            report["ok"] = False
            report["note"] = f"the diagnostic launch failed: {type(ex).__name__}: {ex}"
            return report

        for b in range(sigs.shape[0]):
            if degenerate[b] != 0:
                continue
            for e, name in ((0, "max"), (1, "min")):
                n = int(nk[b, e])
                pos, val = kpos[e][b, :n].copy(), kval[e][b, :n].copy()
                try:
                    ref = reference(Tn, pos, val)
                except Exception:
                    continue  # scipy refuses these knots (PyEMD would raise); nothing to compare
                got = env[e][b]
                if ref.shape != got.shape:
                    continue
                report["checked"] += 1
                if not np.array_equal(ref, got):
                    nd = int(np.count_nonzero(ref != got))
                    report["mismatches"].append(
                        f"signal {b} ({name} envelope, {n} knots): {nd}/{N} samples differ, "
                        f"max |diff| = {float(np.max(np.abs(ref - got))):.3e}")
        report["ok"] = not report["mismatches"]
        if report["checked"] == 0:
            report["note"] = "nothing could be compared (every test signal was degenerate for these settings)"
        if verbose or (warn and not report["ok"]):
            import scipy as _sp
            head = (f"GPUBatchEMD.selftest (spline_kind={kind!r}, extrema_detection={mode!r}, nbsym={self.nbsym}; "
                    f"numpy {np.__version__}, scipy {_sp.__version__}): ")
            if report["ok"]:
                msg = head + f"{report['checked']} envelopes are bit-identical to scipy's."
            else:
                msg = (head + f"{len(report['mismatches'])}/{report['checked']} envelopes differ from scipy's in the last "
                       "bits: " + "; ".join(report["mismatches"][:4]) + ".")
            if verbose and report["ok"]:
                print(msg)
            elif not report["ok"]:
                if warn:
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                if verbose:
                    print(msg)
        return report

    # ------------------------------------------------------------------
    # PyEMD's single-signal interface
    # ------------------------------------------------------------------
    def emd(self, S, T=None, max_imf: int = -1) -> np.ndarray:
        """PyEMD's emd(): decompose ONE 1-D signal and return a numpy array of shape (rows, N): the IMFs, then the
        residue as the last row unless it is all-close to zero. Also sets .imfs (the IMFs without the residue
        row) and .residue, which get_imfs_and_residue() / get_imfs_and_trend() read.  Raises where PyEMD raises."""
        S_host = np.asarray(cp.asnumpy(S) if isinstance(S, cp.ndarray) else S, dtype=np.float64)
        if S_host.ndim != 1:
            raise ValueError(f"emd() takes one 1-D signal, got shape {S_host.shape}; use the batch call for more")
        if T is not None and len(S_host) != len(T):
            raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S_host), len(T)))
        imfs, n_imfs, errors = self.batch(cp.asarray(S_host)[None, :], T, max_imf)
        raise_for_error(int(errors[0]))
        rows = int(n_imfs[0])
        n_real = int(self.last_imf_counts[0])
        out = cp.asnumpy(imfs[0, :rows])
        self.imfs = out[:n_real].copy()
        self.residue = S_host - np.sum(self.imfs, axis=0)  # PyEMD: self.residue = S - np.sum(self.imfs, axis=0)
        return out

    def emd_many(self, S, T=None, max_imf: int = -1) -> list:
        """Decompose every row of a 2-D array in ONE GPU call; returns a list of per-signal numpy arrays, each exactly
        what emd() returns for that row.  Raises (like a loop of PyEMD emd() calls would) for the first row PyEMD
        cannot decompose."""
        S_host = np.asarray(cp.asnumpy(S) if isinstance(S, cp.ndarray) else S, dtype=np.float64)
        if S_host.ndim != 2:
            raise ValueError(f"emd_many() takes a 2-D array, got shape {S_host.shape}")
        imfs, n_imfs, errors = self.batch(cp.asarray(S_host), T, max_imf)
        err = cp.asnumpy(errors)
        for i in np.nonzero(err)[0][:1]:
            raise_for_error(int(err[i]), f"signal {int(i)}")
        n_h = cp.asnumpy(n_imfs)
        host = cp.asnumpy(imfs)
        return [host[i, :int(n_h[i])].copy() for i in range(S_host.shape[0])]

    def get_imfs_and_residue(self):
        """PyEMD's: the IMFs (without the residue row) and the residue of the last emd() call."""
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.imfs, self.residue

    def get_imfs_and_trend(self):
        """PyEMD's: the IMFs and the trend. If the residue is all-close to zero the last IMF is the trend."""
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        return imfs, residue

    def unpack(self, imfs: "cp.ndarray", n_imfs: "cp.ndarray", i: int) -> np.ndarray:
        """Convenience: signal i's rows of a batch result as a numpy array (n_imfs[i], N) -- PyEMD's emd() output."""
        k = int(n_imfs[i])
        return cp.asnumpy(imfs[i, :k, :])


def _pyemd_twin(obj):
    """A CPU PyEMD.EMD with the same settings (for PyEMD's internal helper methods), or None without PyEMD."""
    try:
        from PyEMD import EMD as _CPU
    except Exception:
        return None
    twin = _CPU(spline_kind=str(obj.spline_kind), nbsym=int(obj.nbsym), energy_ratio_thr=obj.energy_ratio_thr,
                std_thr=obj.std_thr, svar_thr=obj.svar_thr, total_power_thr=obj.total_power_thr,
                range_thr=obj.range_thr, extrema_detection=obj.extrema_detection, FIXE=obj.FIXE, FIXE_H=obj.FIXE_H,
                MAX_ITERATION=obj.MAX_ITERATION)
    return twin


class EMD(GPUBatchEMD):
    """Drop-in replacement for PyEMD.EMD (see the module docstring)."""

    def __call__(self, S, T=None, max_imf: int = -1):
        """PyEMD's EMD()(S, T, max_imf) = emd(S, T, max_imf).  A 2-D S is decomposed row by row in one GPU call and a
        list of per-row results is returned."""
        ndim = S.ndim if hasattr(S, "ndim") else np.ndim(S)
        if ndim == 2:
            return self.emd_many(S, T, max_imf)
        return self.emd(S, T=T, max_imf=max_imf)

    def __getattr__(self, name):
        # PyEMD's helper methods (find_extrema, spline_points, check_imf, ...) on the CPU, with this object's settings
        if name.startswith("_") or name in ("imfs", "residue") or "_config" not in self.__dict__:
            raise AttributeError(name)
        twin = _pyemd_twin(self)
        if twin is None or not hasattr(twin, name):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(twin, name)


def as_gpu_emd(ext_EMD):
    """ext_EMD as a GPUBatchEMD: a GPUBatchEMD is used as is; a PyEMD.EMD (CPU) is converted, setting for setting, so
    that `EEMD(ext_EMD=PyEMD.EMD(...))` keeps working after the import is switched."""
    if isinstance(ext_EMD, GPUBatchEMD):
        return ext_EMD
    needed = ("spline_kind", "nbsym", "std_thr", "svar_thr", "energy_ratio_thr", "total_power_thr", "range_thr",
              "extrema_detection", "FIXE", "FIXE_H", "MAX_ITERATION")
    if all(hasattr(ext_EMD, a) for a in needed):
        return EMD(spline_kind=ext_EMD.spline_kind, nbsym=ext_EMD.nbsym, std_thr=ext_EMD.std_thr,
                   svar_thr=ext_EMD.svar_thr, energy_ratio_thr=ext_EMD.energy_ratio_thr,
                   total_power_thr=ext_EMD.total_power_thr, range_thr=ext_EMD.range_thr,
                   extrema_detection=ext_EMD.extrema_detection, FIXE=ext_EMD.FIXE, FIXE_H=ext_EMD.FIXE_H,
                   MAX_ITERATION=ext_EMD.MAX_ITERATION)
    raise TypeError(f"ext_EMD must be a GPUBatchEMD / EMD or a PyEMD.EMD, got {type(ext_EMD)}")

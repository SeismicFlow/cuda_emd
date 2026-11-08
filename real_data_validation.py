#!/usr/bin/env python3
"""
real_data_validation.py
=======================
Does the GPU implementation reproduce PyEMD on real, noisy data?

The script loads a real recording (default: a 5-minute ECG from the MIT-BIH Arrhythmia Database that
ships with SciPy), cuts it into windows of N samples, decomposes the windows with PyEMD (reference, CPU)
and with the GPU library, and compares the two outputs row by row:

    EMD       every window; the GPU receives all windows in one batched call
    EEMD      a few windows; both implementations get the same noise seed
    CEEMDAN   a few windows; both implementations get the same noise seed

Outputs, written to --out (default ./real_data_results):
    real_<name>_combined.pdf/.png    figure for the manuscript: input, first IMFs and residue, the three
                                     algorithms side by side, PyEMD solid, GPU dashed, max|Delta| per panel
    real_<name>_<algo>_full.pdf/.png every IMF of one algorithm (same style as the old Fig. 3)
    agreement_<name>.csv/.txt        the agreement table (IMF counts, max abs error, RMSE, ...)
    run_<name>.json                  machine-readable log incl. GPU / driver / library versions

Usage
-----
    python real_data_validation.py                   # ECG, EMD + EEMD + CEEMDAN
    python real_data_validation.py --only emd        # skip the (slow, CPU-side) ensembles
    python real_data_validation.py --selftest        # no GPU needed: PyEMD stands in for the GPU
    python real_data_validation.py --file x.csv --column 1 --delimiter ,    # your own recording

GPU side - the only part that touches your library
--------------------------------------------------
Your classes are found automatically (flat files next to this script, or the packages cuda_emd /
emd_cuda). If your names differ, point to them:

    --emd-class     cuda_emd.emd:GPUBatchEMD
    --eemd-class    mypkg.eemd:GPUBatchEEMD
    --ceemdan-class mypkg.ceemdan:GPUBatchCEEMDAN

Interfaces that are expected
    EMD      gpu = Cls();  imfs, n_imfs, overflow = gpu(signals[batch, N]);  gpu.unpack(imfs, n_imfs, i)
             (a PyEMD-style class, EMD()(signal) -> rows, is accepted as well)
    EEMD     obj = Cls(trials=..., noise_width=..., [parallel=False]);  obj.noise_seed(seed);
             obj.eemd(signal) or obj(signal)  ->  (rows, N) array, last row = residue (PyEMD convention)
    CEEMDAN  obj = Cls(trials=..., epsilon=...,     [parallel=False]);  obj.noise_seed(seed);
             obj.ceemdan(signal) or obj(signal)  ->  (rows, N) array
If your EMD interface differs, edit Candidate.emd() below; if your EEMD / CEEMDAN differ, edit
_run_ensemble(). Nothing else in the script touches your library.

Data
----
Default dataset: scipy.datasets.electrocardiogram() - excerpt 19:35-24:35 of record 208 (lead MLII) of the
MIT-BIH Arrhythmia Database, 360 Hz, 108,000 samples, in mV. SciPy converts the 11-bit ADC counts with
(adc - 1024) / 200, so the samples lie on a 0.005 mV grid: flat plateaus and exact zeros occur naturally,
which are exactly the cases the extrema / zero-crossing rules of the reference deal with. Per the SciPy
documentation the excerpt includes noise-induced artefacts (60 Hz mains interference among them), typical
heartbeats and premature ventricular contractions; slow baseline drift is visible in the plots.
SciPy >= 1.10 downloads the file once from GitHub and needs `pip install pooch`.
Licence: Open Data Commons Attribution 1.0. PhysioNet also asks for its own standard citation - copy the
current text from https://physionet.org/content/mitdb/1.0.0/ . Please cite (printed again when the script
runs):
    Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag 20(3):45-50,
        2001. doi:10.1109/51.932724
    Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220, 2000.
        doi:10.1161/01.CIR.101.23.e215
    Moody G, Mark R. MIT-BIH Arrhythmia Database (version 1.0.0). PhysioNet, 2005. doi:10.13026/C2F305
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore", category=SyntaxWarning)    # PyEMD docstrings trigger these on first import

ALGOS = ("emd", "eemd", "ceemdan")
TITLES = {"emd": "EMD", "eemd": "EEMD", "ceemdan": "CEEMDAN"}

# Figure colours: slots 1 and 2 of a colour-blind-validated palette. PyEMD is also drawn thick and solid,
# the GPU thin and dashed, so the two stay distinguishable in grey-scale print.
COL_REF, COL_GPU, COL_INK, COL_MUTED, COL_ALERT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d03b3b"
DASH = (0, (3.0, 2.0))

ECG_SOURCE = (
    "MIT-BIH Arrhythmia Database, record 208 (lead MLII), excerpt 19:35-24:35, 360 Hz, 11-bit ADC, via\n"
    "  scipy.datasets.electrocardiogram().  Licence: Open Data Commons Attribution 1.0.  Please cite:\n"
    "  Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag 20(3):45-50,\n"
    "    2001. doi:10.1109/51.932724\n"
    "  Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23):e215-e220, 2000.\n"
    "    doi:10.1161/01.CIR.101.23.e215\n"
    "  Moody G, Mark R. MIT-BIH Arrhythmia Database (version 1.0.0). PhysioNet, 2005. doi:10.13026/C2F305"
)


# ======================================================================================================
# 1. Data
# ======================================================================================================
@dataclass
class Dataset:
    name: str            # short id used in file names
    title: str           # label inside the figures
    x: np.ndarray        # 1-D float64 series
    fs: float = 1.0      # samples per time unit
    t_unit: str = "sample"
    y_unit: str = "a.u."
    source: str = ""


def load_ecg() -> Dataset:
    try:
        try:
            from scipy.datasets import electrocardiogram       # SciPy >= 1.10 (needs `pip install pooch`)
        except ImportError:
            from scipy.misc import electrocardiogram           # SciPy < 1.10 (bundled)
        x = np.asarray(electrocardiogram(), dtype=np.float64)
    except Exception as exc:                                   # offline, pooch missing, ...
        sys.exit("Could not load SciPy's ECG dataset (%s: %s).\n"
                 "  SciPy >= 1.10 downloads it once from GitHub and needs:  pip install pooch\n"
                 "  Alternatively analyse any recording with:  --file data.csv --column 0"
                 % (type(exc).__name__, exc))
    return Dataset("ecg", "ECG, MIT-BIH record 208 (MLII)", x, fs=360.0, t_unit="s", y_unit="mV",
                   source=ECG_SOURCE)


def load_file(a) -> Dataset:
    delim = {"tab": "\t", "semicolon": ";", "comma": ",", "whitespace": None}.get(a.delimiter, a.delimiter)
    try:
        raw = np.genfromtxt(a.file, delimiter=delim, skip_header=a.skiprows, usecols=a.column,
                            dtype=np.float64)
    except Exception as exc:
        sys.exit("Could not read %s (%s: %s)" % (a.file, type(exc).__name__, exc))
    x = np.array(raw, dtype=np.float64).ravel()
    if a.missing is not None:
        x[x == a.missing] = np.nan
    bad = ~np.isfinite(x)
    if bad.all():
        sys.exit("No numeric samples found in column %d of %s." % (a.column, a.file))
    if bad.any():
        idx = np.arange(x.size)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        print("  note: %d missing / non-finite samples were filled by linear interpolation" % bad.sum())
    stem = os.path.splitext(os.path.basename(a.file))[0]
    return Dataset(stem, stem, x, fs=a.fs, t_unit=a.time_unit, y_unit=a.unit,
                   source="user file %s, column %d" % (a.file, a.column))


def make_windows(x: np.ndarray, n: int) -> np.ndarray:
    k = x.size // n
    if k < 1:
        sys.exit("The series has %d samples, fewer than one window (--n %d)." % (x.size, n))
    return np.array(x[: k * n].reshape(k, n), dtype=np.float64, order="C")


def describe(ds: Dataset, W: np.ndarray) -> None:
    x = ds.x
    rep = 100.0 * np.mean(np.diff(W, axis=1) == 0, axis=1)     # % of samples equal to their predecessor
    zer = np.sum(W == 0, axis=1)
    u = np.unique(x)
    step = float(np.diff(u).min()) if u.size > 1 else float("nan")
    print("  signal       : %d samples at %g per %s (%.1f %s), range %.4g .. %.4g %s"
          % (x.size, ds.fs, ds.t_unit, x.size / ds.fs, ds.t_unit, x.min(), x.max(), ds.y_unit))
    print("  windows      : %d x %d samples" % W.shape)
    print("  plateaus     : %.1f-%.1f %% of the samples repeat their predecessor (median %.1f %%)"
          % (rep.min(), rep.max(), np.median(rep)))
    print("  exact zeros  : %d-%d per window (median %.1f)" % (zer.min(), zer.max(), np.median(zer)))
    print("  quantisation : smallest step between distinct values %.4g %s (%d distinct values)"
          % (step, ds.y_unit, u.size))


# ======================================================================================================
# 2. The two implementations
# ======================================================================================================
_CUPY = None


def _split_kwargs(cls, kw):
    """(accepted, rejected) keyword arguments for constructing cls."""
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):                            # not introspectable: pass everything
        return dict(kw), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kw), []
    return {k: v for k, v in kw.items() if k in params}, [k for k in kw if k not in params]


def _cupy():
    """CuPy module or None."""
    global _CUPY
    if _CUPY is None:
        try:
            import cupy
            _CUPY = cupy
        except Exception:
            _CUPY = False
    return _CUPY or None


def _sync() -> None:
    cp = _cupy()
    if cp is not None:
        cp.cuda.Device().synchronize()


def _to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):                                      # CuPy array
        return a.get()
    return np.asarray(a)


def _as_rows(out) -> np.ndarray:
    """Whatever a decomposition call returned -> (rows, N) float64 NumPy array."""
    if isinstance(out, tuple):                                 # (imfs, n_imfs[, overflow]) of a batch call
        imfs, n_imfs = _to_numpy(out[0]), _to_numpy(out[1])
        return np.array(imfs[0, : int(n_imfs[0])], dtype=np.float64)
    a = _to_numpy(out)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError("expected a (rows, N) array, got shape %s" % (a.shape,))
    return np.array(a, dtype=np.float64)


def _unpack(gpu, imfs, n_imfs, i: int) -> np.ndarray:
    try:
        rows = gpu.unpack(imfs, n_imfs, i)
    except TypeError:                                          # other signature: trim by hand
        rows = _to_numpy(imfs)[i, : int(_to_numpy(n_imfs)[i])]
    return np.array(_to_numpy(rows), dtype=np.float64)


def _run_ensemble(cls, algo: str, x: np.ndarray, seed: int, kw: dict) -> np.ndarray:
    """EEMD / CEEMDAN through the PyEMD protocol (see module docstring)."""
    kw, _ = _split_kwargs(cls, kw)                             # arguments the class does not take are left out
    try:
        if "seed" in inspect.signature(cls).parameters:
            kw.setdefault("seed", seed)
    except (TypeError, ValueError):
        pass
    obj = cls(**kw)
    if hasattr(obj, "noise_seed"):
        obj.noise_seed(seed)
    fn = getattr(obj, algo, None) or obj                       # obj.eemd(x) / obj.ceemdan(x) / obj(x)
    out = fn(x)
    _sync()
    return _as_rows(out)


class Reference:
    """PyEMD, default configuration, one signal per call.

    EEMD / CEEMDAN run with parallel=False. PyEMD >= 1.8 defaults to parallel=True, which draws the noise
    inside the worker processes: with the same seed it yields different noise realisations than the serial
    run (e.g. a different number of output rows), so it cannot be the bit-exact reference for an
    implementation that reproduces the serial noise stream."""

    def __init__(self):
        try:
            from PyEMD import EMD, EEMD, CEEMDAN
        except ImportError:
            sys.exit("PyEMD is required:  pip install EMD-signal")
        self._cls = {"emd": EMD, "eemd": EEMD, "ceemdan": CEEMDAN}

    def emd(self, windows):
        t0 = time.perf_counter()
        rows = [np.array(self._cls["emd"]()(w), dtype=np.float64) for w in windows]
        return rows, time.perf_counter() - t0, None

    def ensemble(self, algo, x, seed, trials, param):
        kw = {"trials": trials, "noise_width" if algo == "eemd" else "epsilon": param, "parallel": False}
        t0 = time.perf_counter()
        out = _run_ensemble(self._cls[algo], algo, x, seed, kw)
        return out, time.perf_counter() - t0


class Candidate:
    """The GPU library. Everything that touches your code is in this class and in _run_ensemble()."""

    def __init__(self, classes: dict):
        self.cls = classes
        self._warm = set()

    def has(self, algo: str) -> bool:
        return self.cls.get(algo) is not None

    def emd(self, windows):
        gpu = self.cls["emd"]()
        sig = np.ascontiguousarray(windows, dtype=np.float64)
        batch = hasattr(gpu, "unpack")             # GPUBatchEMD protocol, else PyEMD-style EMD()(signal)
        # Untimed warm-up: the first call compiles the kernels (NVRTC).
        if batch and hasattr(gpu, "warmup"):
            gpu.warmup()
        else:
            gpu(sig[:1]) if batch else gpu(sig[0])
        _sync()
        # Timed interval: input transfer + kernel + synchronisation + unpacking to NumPy.
        t0 = time.perf_counter()
        flags = None
        if batch:
            res = gpu(sig)
            imfs, n_imfs, overflow = res[0], res[1], res[2]
            _sync()
            rows = [_unpack(gpu, imfs, n_imfs, i) for i in range(len(sig))]
            flags = [bool(v) for v in np.ravel(_to_numpy(overflow))]
        else:
            rows = [_as_rows(gpu(w)) for w in sig]
            _sync()
        return rows, time.perf_counter() - t0, flags

    def ensemble(self, algo, x, seed, trials, param):
        key = "noise_width" if algo == "eemd" else "epsilon"
        cls = self.cls[algo]
        # GPUBatchEEMD / GPUBatchCEEMDAN default to noise_backend="cupy" (noise drawn on the GPU: a different
        # random stream than PyEMD's numpy RandomState), so with the same seed they can NEVER match PyEMD
        # bit for bit. For this comparison the candidate must use PyEMD's stream ("numpy"); classes without
        # that argument get it filtered out by _split_kwargs. PyEMD's own classes are left untouched.
        extra = {} if cls.__module__.split(".")[0] == "PyEMD" else {"noise_backend": "numpy"}
        if algo not in self._warm:                  # untimed warm-up with a tiny ensemble
            _run_ensemble(cls, algo, x, seed, {"trials": 2, key: param, "parallel": False, **extra})
            self._warm.add(algo)
        t0 = time.perf_counter()
        out = _run_ensemble(cls, algo, x, seed, {"trials": trials, key: param, "parallel": False, **extra})
        return out, time.perf_counter() - t0


_MODULES = ("cuda_emd.emd", "emd_cuda.emd", "cuda_emd", "emd_cuda", "emd", "eemd", "ceemdan")
_NAMES = {"emd": ("GPUBatchEMD", "EMD"),
          "eemd": ("GPUBatchEEMD", "GPUEEMD", "EEMD"),
          "ceemdan": ("GPUBatchCEEMDAN", "GPUCEEMDAN", "CEEMDAN")}


def find_class(algo: str, override, problems: list):
    if override:
        mod, _, name = override.partition(":")
        if not name:
            sys.exit("--%s-class expects module:ClassName, got %r" % (algo, override))
        return getattr(importlib.import_module(mod), name)
    for m in _MODULES:
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError as exc:
            # Silent only when the candidate module itself (or its parent package) is absent. A missing
            # DEPENDENCY of an existing module (kernels.py, cupy, ...) is a real problem and must be shown.
            if exc.name and (m == exc.name or m.startswith(exc.name + ".")):
                continue
            problems.append("%s: %s: %s" % (m, type(exc).__name__, exc))
            continue
        except Exception as exc:                               # module exists but fails to import
            problems.append("%s: %s: %s" % (m, type(exc).__name__, exc))
            continue
        for cname in _NAMES[algo]:
            if hasattr(mod, cname):
                return getattr(mod, cname)
    return None


def build_candidate(a, wanted):
    if a.selftest:
        from PyEMD import EMD, EEMD, CEEMDAN
        return Candidate({"emd": EMD, "eemd": EEMD, "ceemdan": CEEMDAN}), "GPU stand-in (= PyEMD)"
    for p in a.gpu_path:
        sys.path.insert(0, os.path.abspath(p))
    classes, problems = {}, []
    for algo in wanted:
        try:
            classes[algo] = find_class(algo, getattr(a, algo + "_class"), problems)
        except Exception as exc:
            sys.exit("--%s-class could not be imported: %s: %s" % (algo, type(exc).__name__, exc))
    for algo in wanted:
        if classes[algo] is None:
            print("  ! no GPU class found for %s (tried modules %s, names %s) - skipped.\n"
                  "    Point to it with --%s-class module:ClassName"
                  % (TITLES[algo], ", ".join(_MODULES), ", ".join(_NAMES[algo]), algo))
    for p in dict.fromkeys(problems):                          # each problem once, in order found
        print("  ! import problem while searching: " + p)
    for algo in wanted:
        c = classes.get(algo)
        if c is None:
            continue
        if c.__module__.split(".")[0] == "PyEMD":              # e.g. a module that re-exports PyEMD's class
            sys.exit("\n! The class found for %s is PyEMD's own (%s.%s). Comparing PyEMD with itself proves "
                     "nothing.\n  Point to your GPU class with --%s-class module:ClassName"
                     % (TITLES[algo], c.__module__, c.__name__, algo))
        print("  GPU class for %-7s: %s.%s" % (TITLES[algo], c.__module__, c.__name__))
        if algo != "emd":                                      # settings that PyEMD gets must reach the GPU class
            want = {"trials": a.trials, "noise_width": a.noise_width} if algo == "eemd" else \
                   {"trials": a.trials, "epsilon": a.epsilon}
            for k in _split_kwargs(c, want)[1]:
                print("  ! %s has no argument '%s': it runs with its own default, while PyEMD uses %s = %g. "
                      "Make sure the two are equal." % (c.__name__, k, k, want[k]))
    return Candidate(classes), "GPU (this work)"


# ======================================================================================================
# 3. Comparison
# ======================================================================================================
def compare(ref: np.ndarray, gpu: np.ndarray, x: np.ndarray, tol: float, flagged: bool) -> dict:
    s = {"rows_ref": int(ref.shape[0]), "rows_gpu": int(gpu.shape[0]), "same_shape": ref.shape == gpu.shape,
         "flagged": bool(flagged)}
    if s["same_shape"]:
        d = np.abs(gpu - ref)
        rng = float(x.max() - x.min()) or 1.0
        s.update(max_abs=float(d.max()), sse=float(np.sum(d * d)), count=int(d.size),
                 per_row=[float(v) for v in d.max(axis=1)], bitwise=bool(np.array_equal(gpu, ref)),
                 max_rel=float(d.max()) / rng)
    else:
        s.update(max_abs=float("nan"), sse=float("nan"), count=0, per_row=None, bitwise=False,
                 max_rel=float("nan"))
    s["ok"] = bool(s["same_shape"] and np.isfinite(s["max_abs"]) and s["max_abs"] < tol and not flagged)
    return s


def run_algorithm(algo, ref, cand, W, ids, a) -> dict:
    wins = W[ids]
    if algo == "emd":
        print("  PyEMD : %d windows, one signal per call ..." % len(ids), flush=True)
        ref_rows, t_ref, _ = ref.emd(wins)
        print("          %.1f s\n  GPU   : %d windows, one batched call ..." % (t_ref, len(ids)), flush=True)
        gpu_rows, t_gpu, flags = cand.emd(wins)
        print("          %.2f s" % t_gpu, flush=True)
        flags = flags or [False] * len(ids)
    else:
        param = a.noise_width if algo == "eemd" else a.epsilon
        ref_rows, gpu_rows, t_ref, t_gpu, flags = [], [], 0.0, 0.0, [False] * len(ids)
        for j, wid in enumerate(ids):
            seed = a.seed + wid
            print("  window %d (seed %d): PyEMD ..." % (wid, seed), end="", flush=True)
            r, tr = ref.ensemble(algo, wins[j], seed, a.trials, param)
            print(" %.1f s, GPU ..." % tr, end="", flush=True)
            g, tg = cand.ensemble(algo, wins[j], seed, a.trials, param)
            print(" %.2f s" % tg, flush=True)
            ref_rows.append(r)
            gpu_rows.append(g)
            t_ref += tr
            t_gpu += tg
    stats = [compare(r, g, wins[j], a.tol, flags[j]) for j, (r, g) in enumerate(zip(ref_rows, gpu_rows))]
    return {"ids": [int(i) for i in ids], "ref": ref_rows, "gpu": gpu_rows, "stats": stats,
            "t_ref": t_ref, "t_gpu": t_gpu}


def summarise(algo: str, r: dict) -> dict:
    st = r["stats"]
    matched = [s for s in st if s["same_shape"]]
    cnt = sum(s["count"] for s in matched)
    nan = float("nan")
    return {
        "algorithm": TITLES[algo],
        "windows": len(st),
        "rows_min": int(min(s["rows_ref"] for s in st)),
        "rows_max": int(max(s["rows_ref"] for s in st)),
        "same_imf_count": sum(s["same_shape"] for s in st),
        "bitwise_identical": sum(s["bitwise"] for s in st),
        "within_tolerance": sum(s["ok"] for s in st),
        "max_abs_error": float(np.max([s["max_abs"] for s in matched])) if matched else nan,
        "rmse": (sum(s["sse"] for s in matched) / cnt) ** 0.5 if cnt else nan,
        "max_error_over_range": float(np.max([s["max_rel"] for s in matched])) if matched else nan,
        "overflow_flagged": sum(s["flagged"] for s in st),
        "pyemd_seconds": r["t_ref"],
        "gpu_seconds": r["t_gpu"],
    }


def _num(v, spec="%.2e"):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else spec % v


def table_lines(rows, unit, a, ds) -> list:
    head = ["algorithm", "windows", "rows (PyEMD)", "same row count", "bitwise identical", "within tol.",
            "max abs err [%s]" % unit, "RMSE [%s]" % unit, "max err / range", "overflow"]
    body = [[r["algorithm"], str(r["windows"]),
             str(r["rows_min"]) if r["rows_min"] == r["rows_max"] else "%d-%d" % (r["rows_min"], r["rows_max"]),
             "%d/%d" % (r["same_imf_count"], r["windows"]),
             "%d/%d" % (r["bitwise_identical"], r["windows"]), "%d/%d" % (r["within_tolerance"], r["windows"]),
             _num(r["max_abs_error"]), _num(r["rmse"]), _num(r["max_error_over_range"]),
             str(r["overflow_flagged"])]
            for r in rows]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    line = lambda cells: "  ".join(c.ljust(w[i]) for i, c in enumerate(cells))
    out = ["Agreement of the GPU implementation with PyEMD on real data: %s, N = %d, tolerance %g %s"
           % (ds.title, a.n, a.tol, unit), "", line(head), line(["-" * x for x in w])]
    out += [line(b) for b in body]
    out += ["",
            "Rows = IMFs plus the residue (last row). Errors are absolute, max |GPU - PyEMD| over all rows and "
            "samples, in %s;" % unit,
            "'max err / range' is the same error divided by the range (max - min) of the input window.",
            "'within tol.' = same row count, max abs error < tolerance and no overflow flag.",
            "'overflow' = windows for which the GPU overflow flag was set.",
            "(Run times are in the CSV / JSON only: 52 windows are not a benchmark batch.)"]
    return out


# ======================================================================================================
# 4. Figures
# ======================================================================================================
def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",       # keep text as text in vector files
    })


def _delta_text(d):
    if d is None or not np.isfinite(d):
        return None
    return r"$\max|\Delta|$ = 0" if d == 0 else r"$\max|\Delta|$ = %.2e" % d


def _panel(ax, t, ref, gpu, delta, ylabel=None, alert=None) -> None:
    ax.plot(t, ref, color=COL_REF, lw=1.3, zorder=2)
    if gpu is not None:
        ax.plot(t, gpu, color=COL_GPU, lw=0.8, ls=DASH, zorder=3)
    lo, hi = float(np.min(ref)), float(np.max(ref))
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.08 * span, hi + 0.34 * span)                  # head-room for the annotation
    text, colour = (alert, COL_ALERT) if alert else (_delta_text(delta), COL_MUTED)
    if text:
        ax.text(0.99, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=colour)
    if ylabel:
        ax.set_ylabel(ylabel)


def _input_panel(fig, cell, ds, t, x, caption):
    """The raw window; the caption sits in head-room above the trace so it never covers the signal."""
    ax = fig.add_subplot(cell)
    ax.plot(t, x, color=COL_INK, lw=0.7)
    lo, hi = float(x.min()), float(x.max())
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.05 * span, hi + 0.30 * span)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Input (%s)" % ds.y_unit)
    ax.tick_params(labelbottom=False, bottom=False)
    ax.text(0.005, 0.97, caption, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=COL_MUTED)
    return ax


def _legend_row(fig, gs_cell, label):
    from matplotlib.lines import Line2D
    ax = fig.add_subplot(gs_cell)
    ax.axis("off")
    handles = [Line2D([], [], color=COL_REF, lw=1.3), Line2D([], [], color=COL_GPU, lw=0.8, ls=DASH)]
    ax.legend(handles, ["PyEMD (reference, CPU)", label], loc="center", ncol=2, frameon=False, fontsize=7,
              handlelength=2.6)


def _save(fig, base, selftest) -> list:
    if selftest:
        fig.text(0.5, 0.008, "SELFTEST - the GPU curve is PyEMD itself; do not use this figure in the manuscript",
                 ha="center", color=COL_ALERT, fontsize=6.5)
    import matplotlib.pyplot as plt
    paths = []
    for ext in ("pdf", "png"):
        paths.append("%s.%s" % (base, ext))
        fig.savefig(paths[-1], dpi=300)
    plt.close(fig)
    return paths


def _row_content(st, ref, gpu, idx):
    """(gpu row or None, max|Delta| or None, alert text or None) for row idx of one comparison."""
    if st["same_shape"]:
        return gpu[idx], st["per_row"][idx], None
    return None, None, "IMF count differs: PyEMD %d, GPU %d" % (st["rows_ref"], st["rows_gpu"])


def fig_combined(ds, wid, res, algos, a, label, base) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _style()
    K, ncol = a.show_imfs, len(algos)
    n = a.n
    t = (wid * n + np.arange(n)) / ds.fs
    x = ds.x[wid * n:(wid + 1) * n]
    height = 1.65 + 0.88 * (K + 1)
    fig = plt.figure(figsize=({1: 3.6, 2: 5.6, 3: 7.48}[ncol], height))
    gs = GridSpec(K + 4, ncol, figure=fig, height_ratios=[0.22, 1.15, 0.24] + [1.0] * (K + 1),
                  left=0.095 if ncol > 1 else 0.22, right=0.99, top=0.99, bottom=0.62 / height,
                  hspace=0.22, wspace=0.30)
    _legend_row(fig, gs[0, :], label)
    ax_in = _input_panel(fig, gs[1, :], ds, t, x, "%s, N = %d" % (ds.title, n))
    col0 = [ax_in]
    for c, algo in enumerate(algos):
        r = res[algo]
        k = r["ids"].index(wid)
        ref, gpu, st = r["ref"][k], r["gpu"][k], r["stats"][k]
        n_rows = ref.shape[0]
        for rr in range(K + 1):
            ax = fig.add_subplot(gs[3 + rr, c], sharex=ax_in)
            is_res = rr == K
            if not is_res and rr >= n_rows - 1:                      # fewer IMFs than panels
                ax.axis("off")
                continue
            idx = n_rows - 1 if is_res else rr
            g, delta, alert = _row_content(st, ref, gpu, idx)
            lab = None if c else ("residual\ntrend" if is_res else "IMF %d" % (rr + 1))
            _panel(ax, t, ref[idx], g, delta, lab, alert if rr == 0 else None)
            if c == 0:
                col0.append(ax)
            if rr == 0:
                extra = "" if algo == "emd" else " (%d trials)" % a.trials
                ax.set_title(TITLES[algo] + extra, pad=3)
            if is_res:
                ax.set_xlabel("Time (%s)" % ds.t_unit)
            else:
                ax.tick_params(labelbottom=False, bottom=False)
    fig.align_ylabels(col0)
    return _save(fig, base, a.selftest)


def fig_full(ds, wid, algo, r, a, label, base) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _style()
    k = r["ids"].index(wid)
    ref, gpu, st = r["ref"][k], r["gpu"][k], r["stats"][k]
    n_rows, n = ref.shape[0], a.n
    t = (wid * n + np.arange(n)) / ds.fs
    x = ds.x[wid * n:(wid + 1) * n]
    height = 1.45 + 0.8 * n_rows
    fig = plt.figure(figsize=(5.6, height))
    gs = GridSpec(n_rows + 2, 1, figure=fig, height_ratios=[0.22, 1.1] + [1.0] * n_rows, left=0.135,
                  right=0.985, top=0.99, bottom=0.62 / height, hspace=0.22)
    _legend_row(fig, gs[0, 0], label)
    extra = "" if algo == "emd" else ", %d trials" % a.trials
    ax_in = _input_panel(fig, gs[1, 0], ds, t, x, "%s, N = %d - %s%s" % (ds.title, n, TITLES[algo], extra))
    axs = [ax_in]
    for i in range(n_rows):
        ax = fig.add_subplot(gs[2 + i, 0], sharex=ax_in)
        g, delta, alert = _row_content(st, ref, gpu, i)
        _panel(ax, t, ref[i], g, delta, "residual\ntrend" if i == n_rows - 1 else "IMF %d" % (i + 1),
               alert if i == 0 else None)
        axs.append(ax)
        if i == n_rows - 1:
            ax.set_xlabel("Time (%s)" % ds.t_unit)
        else:
            ax.tick_params(labelbottom=False, bottom=False)
    fig.align_ylabels(axs)
    return _save(fig, base, a.selftest)


# ======================================================================================================
# 5. Bookkeeping
# ======================================================================================================
def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor(),
           "cpu_count": os.cpu_count(), "numpy": np.__version__}
    for mod in ("scipy", "matplotlib"):
        try:
            env[mod] = importlib.import_module(mod).__version__
        except Exception:
            env[mod] = "n/a"
    try:
        from importlib.metadata import version
        env["PyEMD (EMD-signal)"] = version("EMD-signal")
    except Exception:
        env["PyEMD (EMD-signal)"] = "n/a"
    cp = _cupy()
    if cp is None:
        env["cupy"] = "n/a"
        return env
    try:
        ver = lambda v: "%d.%d" % (v // 1000, (v % 1000) // 10)
        p = cp.cuda.runtime.getDeviceProperties(0)
        name = p["name"]
        env.update({
            "cupy": cp.__version__,
            "gpu": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": "%d.%d" % (p["major"], p["minor"]),
            "multiprocessors": int(p["multiProcessorCount"]),
            "shared_memory_per_block_bytes": int(p["sharedMemPerBlock"]),
            "device_memory_bytes": int(p["totalGlobalMem"]),
            "cuda_runtime": ver(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": ver(cp.cuda.runtime.driverGetVersion()),
        })
        try:
            env["nvrtc"] = "%d.%d" % tuple(cp.cuda.nvrtc.getVersion())
        except Exception:
            pass
    except Exception as exc:
        env["gpu_query_error"] = "%s: %s" % (type(exc).__name__, exc)
    return env


def pick_windows(n_win: int, k: int, fig_win: int) -> list:
    """The figure window plus k-1 others spread evenly over the record."""
    rest = [i for i in range(n_win) if i != fig_win]
    extra = []
    if k > 1 and rest:
        extra = [rest[int(round(v))] for v in np.linspace(0, len(rest) - 1, min(k - 1, len(rest)))]
    return sorted(set([fig_win] + extra))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--file", help="analyse a text/CSV file instead of the built-in ECG")
    g.add_argument("--column", type=int, default=0, help="0-based column of --file (default 0)")
    g.add_argument("--delimiter", default=",", help="separator of --file: a character, or tab / whitespace")
    g.add_argument("--skiprows", type=int, default=0, help="header lines to skip in --file")
    g.add_argument("--missing", type=float, default=None, help="value in --file that marks a missing sample")
    g.add_argument("--fs", type=float, default=1.0, help="samples per time unit of --file (default 1)")
    g.add_argument("--time-unit", default="sample", help="time-axis label of --file (default 'sample')")
    g.add_argument("--unit", default="a.u.", help="amplitude unit of --file")
    g.add_argument("--n", type=int, default=2048, help="window length (default 2048, as in the paper)")
    g.add_argument("--window", type=int, default=None,
                   help="window shown in the figures. Default 3 for the ECG (t = 17.1-22.8 s: normal beats, "
                        "ventricular extrasystoles, baseline wander, noise), 0 for --file")
    g.add_argument("--max-windows", type=int, default=200, help="use at most this many windows (default 200)")
    g = p.add_argument_group("algorithms")
    g.add_argument("--only", default="emd,eemd,ceemdan", help="comma list of emd,eemd,ceemdan")
    g.add_argument("--trials", type=int, default=100, help="ensemble size of EEMD and CEEMDAN (default 100)")
    g.add_argument("--noise-width", type=float, default=0.05, help="EEMD noise width (PyEMD default 0.05)")
    g.add_argument("--epsilon", type=float, default=0.005, help="CEEMDAN epsilon (PyEMD default 0.005)")
    g.add_argument("--seed", type=int, default=12345, help="base noise seed; window i uses seed + i")
    g.add_argument("--ensemble-windows", type=int, default=3,
                   help="windows for EEMD / CEEMDAN: the figure window plus N-1 others (default 3)")
    g.add_argument("--tol", type=float, default=1e-6, help="absolute tolerance for 'within tol.' (default 1e-6)")
    g = p.add_argument_group("GPU library")
    g.add_argument("--gpu-path", action="append", default=[], metavar="DIR",
                   help="extra folder to import your library from (repeatable)")
    g.add_argument("--emd-class", metavar="MODULE:CLASS")
    g.add_argument("--eemd-class", metavar="MODULE:CLASS")
    g.add_argument("--ceemdan-class", metavar="MODULE:CLASS")
    g.add_argument("--selftest", action="store_true",
                   help="dry run without a GPU: PyEMD stands in for the GPU. Tests the data, comparison, "
                        "table and figure code; the outputs are marked SELFTEST")
    g = p.add_argument_group("output")
    g.add_argument("--out", default="real_data_results", help="output folder")
    g.add_argument("--show-imfs", type=int, default=4, help="IMF rows in the combined figure (default 4)")
    a = p.parse_args(argv)
    a.only = [s.strip().lower() for s in a.only.split(",") if s.strip()]
    bad = [s for s in a.only if s not in ALGOS]
    if bad:
        p.error("--only: unknown algorithm(s) %s (choose from %s)" % (", ".join(bad), ", ".join(ALGOS)))
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(a.out, exist_ok=True)

    print("=" * 100)
    if a.selftest:
        print("SELFTEST: PyEMD stands in for the GPU, so every difference will be exactly 0. This only checks\n"
              "the pipeline (data, comparison, table, figures). Do NOT use these outputs in the manuscript.")
        print("=" * 100)
    ds = load_file(a) if a.file else load_ecg()
    W = make_windows(ds.x, a.n)
    fig_win = a.window if a.window is not None else (3 if not a.file and len(W) > 3 else 0)
    if not 0 <= fig_win < len(W):
        sys.exit("--window %d is out of range (0..%d)" % (fig_win, len(W) - 1))
    W = W[: max(a.max_windows, fig_win + 1)]
    name = ds.name + ("_selftest" if a.selftest else "")
    print("Dataset : %s\nSource  : %s" % (ds.title, ds.source))
    describe(ds, W)
    print("Figure window: %d (%.2f-%.2f %s)" % (fig_win, fig_win * a.n / ds.fs, (fig_win + 1) * a.n / ds.fs,
                                               ds.t_unit))
    print("-" * 100)

    ref = Reference()
    cand, label = build_candidate(a, a.only)
    algos = [k for k in ALGOS if k in a.only and cand.has(k)]
    if not algos:
        sys.exit("\nNo GPU implementation could be loaded. Check the 'import problem' lines above (a missing cupy / "
                 "kernels.py looks like this), fix the class names (see --help), or run --selftest.")

    res, failed = {}, {}
    ens_ids = pick_windows(len(W), a.ensemble_windows, fig_win)
    for algo in algos:
        ids = list(range(len(W))) if algo == "emd" else ens_ids
        print("\n[%s]  %d window(s)%s" % (TITLES[algo], len(ids),
                                         "" if algo == "emd" else ", %d trials, seeds %d + window index"
                                         % (a.trials, a.seed)))
        try:
            res[algo] = run_algorithm(algo, ref, cand, W, ids, a)
        except Exception as exc:
            failed[algo] = "%s: %s" % (type(exc).__name__, exc)
            print("  ! %s failed and is skipped:" % TITLES[algo])
            traceback.print_exc(limit=4)
    algos = [k for k in algos if k in res]
    if not algos:
        sys.exit("\nAll algorithms failed - see the messages above.")

    # ---- table ----------------------------------------------------------------------------------------
    rows = [summarise(k, res[k]) for k in algos]
    lines = table_lines(rows, ds.y_unit, a, ds)
    print("\n" + "=" * 100)
    print("\n".join(lines))
    with open(os.path.join(a.out, "agreement_%s.txt" % name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "agreement_%s.csv" % name), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- figures --------------------------------------------------------------------------------------
    made = []
    try:
        made += fig_combined(ds, fig_win, res, algos, a, label, os.path.join(a.out, "real_%s_combined" % name))
        for algo in algos:
            made += fig_full(ds, fig_win, algo, res[algo], a, label,
                             os.path.join(a.out, "real_%s_%s_full" % (name, algo)))
    except Exception:
        print("\n! Figure generation failed (the table above is still valid):")
        traceback.print_exc(limit=6)

    # ---- log ------------------------------------------------------------------------------------------
    env = environment()
    log = {"args": vars(a), "environment": env, "dataset": {"title": ds.title, "samples": int(ds.x.size),
           "fs": ds.fs, "source": ds.source}, "figure_window": fig_win, "summary": rows, "failed": failed,
           "per_window": {k: [{"window": w, "same_shape": s["same_shape"], "rows_pyemd": s["rows_ref"],
                               "rows_gpu": s["rows_gpu"], "max_abs": s["max_abs"], "bitwise": s["bitwise"],
                               "flagged": s["flagged"], "per_row_max_abs": s["per_row"]}
                              for w, s in zip(res[k]["ids"], res[k]["stats"])] for k in algos}}
    with open(os.path.join(a.out, "run_%s.json" % name), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print("\nEnvironment: " + ", ".join("%s %s" % (k, v) for k, v in env.items()))
    print("\nFiles in %s:" % os.path.abspath(a.out))
    for pth in sorted(set(made)) + [os.path.join(a.out, f) for f in
                                    ("agreement_%s.csv" % name, "agreement_%s.txt" % name, "run_%s.json" % name)]:
        print("  " + os.path.basename(pth))

    n_cmp = sum(r["windows"] for r in rows)
    n_ok = sum(r["within_tolerance"] for r in rows)
    flagged = sum(r["overflow_flagged"] for r in rows)
    print()
    if n_ok == n_cmp and not failed:
        print("RESULT: all %d comparisons agree with PyEMD (same IMF count, max abs error %.2e %s < %g)."
              % (n_cmp, max(r["max_abs_error"] for r in rows), ds.y_unit, a.tol))
        return 0
    print("RESULT: %d of %d comparisons are within tolerance; see the table and the JSON log." % (n_ok, n_cmp))
    for k in algos:
        bad = [w for w, s in zip(res[k]["ids"], res[k]["stats"]) if not s["ok"]]
        if bad:
            print("  %s: window(s) %s%s not within tolerance" % (TITLES[k], bad[:12], " ..." if len(bad) > 12 else ""))
    n_rows = sum(1 for k in algos for s in res[k]["stats"] if not s["same_shape"])
    n_err = sum(1 for k in algos for s in res[k]["stats"] if s["same_shape"] and not s["max_abs"] < a.tol)
    if n_rows:
        print("  %d window(s) differ in the number of rows (IMFs + residue)." % n_rows)
    if n_err:
        print("  %d window(s) have a max abs error >= %g %s." % (n_err, a.tol, ds.y_unit))
    if flagged:
        print("  %d window(s) set the GPU overflow flag (too many extrema for the shared-memory capacity): "
              "try a smaller --n." % flagged)
    if failed:
        print("  skipped algorithms: " + "; ".join("%s (%s)" % kv for kv in failed.items()))
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyemd_profile.py -- reproducible profiling study of PyEMD (pip: ``EMD-signal``)
================================================================================

Profiles the ``EMD``, ``EEMD`` and ``CEEMDAN`` classes of PyEMD and answers:

  * where does the time go, per algorithm stage (stage timers, primary source),
  * how does that change with signal length / signal type,
  * how much of it is interpreter + library-dispatch overhead vs compiled work,
  * what is the Amdahl-law ceiling per stage.

Two complementary measurements are made on the same configurations:

  (a) *Stage timers* (``time.perf_counter_ns``): the library's own methods are
      wrapped (monkeypatched, always restored) with exclusive-time accounting.
      PyEMD has no separate methods for "envelope mean + subtraction" or the
      outer loop bookkeeping, so an instrumented copy of ``EMD.emd`` is used for
      those segments.  Its output is verified to be bit-identical to the
      library's own ``EMD.emd`` before it is used (else the script falls back
      to method-level timers only).  Unprofiled totals are measured separately.
  (b) *cProfile/pstats* for call counts and the compiled-vs-Python split, plus
      optional ``line_profiler`` for the hottest methods.  cProfile adds overhead
      and cannot see NumPy operator kernels (``a - b``) as separate entries, so
      (a) is the primary source for stage fractions.

Typical use
-----------
    pip install -r requirements.txt
    python pyemd_profile.py --mode quick --outdir results_quick     # minutes
    python pyemd_profile.py --mode full --estimate                  # runtime estimate only
    python pyemd_profile.py --mode full --outdir results_full       # hours (see README)
    python pyemd_profile.py --report-only --outdir results_full     # rebuild tables/figures

Everything is deterministic given the seeds (see ``--seed``) apart from the
timings themselves.  All numbers in ``findings_summary.md`` are computed from
the raw data written by this script.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import zlib
from collections import OrderedDict, defaultdict
from time import perf_counter_ns

SCRIPT_VERSION = "1.0.1"
LOG = logging.getLogger("pyemd_profile")

# --------------------------------------------------------------------------- #
# Experimental constants                                                      #
# --------------------------------------------------------------------------- #
SIGNALS = ("multitone", "multitone_noisy", "chirp_am")
SIGNAL_LABEL = {
    "multitone": "multi-tone",
    "multitone_noisy": "multi-tone + noise (10 dB)",
    "chirp_am": "chirp + AM",
}
FS_HZ = 1000.0                                   # sampling rate of all test signals
TONES_HZ = (2.0, 6.0, 18.0, 54.0, 162.0)         # multi-tone components
TONES_AMP = (1.0, 0.8, 0.6, 0.5, 0.4)
SNR_DB = 10.0                                    # for "multitone_noisy"
CHIRP = {"f0": 5.0, "f1": 200.0, "f_am": 2.0, "am_depth": 0.5}
FULL_LENGTHS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
FLOOR_N = 64                                     # tiny signal for per-call overhead floors

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")

# Fine-grained timer names -> the stage groups that are reported in the paper.
FINE_TO_GROUP = {
    "extrema_detection": "extrema",
    "boundary_handling": "boundary",
    "spline_construction": "spline_build",
    "tridiagonal_solve": "spline_build",
    "spline_3pt": "spline_build",
    "spline_grid_select": "spline_eval",
    "spline_evaluation": "spline_eval",
    "envelope_mean_subtract": "mean_subtract",
    "stop_extrema": "stop_check",
    "stop_check": "stop_check",
    "outer_loop": "outer_loop",
    "end_condition": "outer_loop",
    "extract_glue": "outer_loop",
    # ensemble level (EEMD / CEEMDAN)
    "noise_generation": "noise_generation",
    "ensemble_glue": "ensemble_glue",
    "trial_update": "ensemble_glue",     # per-trial wrapper; its inclusive time = the work a Pool would parallelise
}
EMD_GROUPS = ["extrema", "boundary", "spline_build", "spline_eval",
              "mean_subtract", "stop_check", "outer_loop"]
ENS_GROUPS = ["noise_generation", "ensemble_glue"]
GROUP_LABEL = {
    "extrema": "Extrema detection",
    "boundary": "Boundary / mirror handling",
    "spline_build": "Spline construction (incl. tridiagonal solve)",
    "spline_eval": "Spline evaluation",
    "mean_subtract": "Envelope mean and subtraction",
    "stop_check": "Stopping criterion (IMF check)",
    "outer_loop": "Outer loop, IMF bookkeeping, end condition",
    "noise_generation": "Noise generation",
    "ensemble_glue": "Ensemble aggregation / glue",
}
GROUP_SHORT = {
    "extrema": "extrema", "boundary": "boundary", "spline_build": "spline build",
    "spline_eval": "spline eval", "mean_subtract": "mean/subtract",
    "stop_check": "stop check", "outer_loop": "loop/bookkeeping",
    "noise_generation": "noise gen.", "ensemble_glue": "ensemble glue",
}

# Amdahl partition (finer than EMD_GROUPS: the tridiagonal solve is split out of spline construction).
# (key, label, fine timers, class, one-sentence justification from the source)
#   DP    = data-parallel as written (independent per sample / per interval, or a plain reduction)
#   MIXED = partly sequential as written, but parallelisable with restructuring or a different algorithm
#   SEQ   = inherently sequential (loop-carried dependency)
AMDAHL_STAGES = [
    ("extrema", "Extrema detection", ("extrema_detection",), "DP",
     "Element-wise first differences and sign tests followed by index compaction (nonzero); every sample is independent, i.e. a map plus a stream-compaction scan."),
    ("boundary", "Boundary / mirror handling", ("boundary_handling",), "MIXED",
     "Scalar branching on the first and last extrema and mirroring of O(nbsym) points is sequential, but the stage also gathers, concatenates and de-duplicates O(K) extrema arrays, which are data-parallel copies and compaction."),
    ("spline_assembly", "Spline coefficient assembly", ("spline_construction", "spline_3pt"), "DP",
     "Per-interval divided differences and coefficient assembly in CubicSpline (plus input validation) are independent across intervals."),
    ("tridiagonal", "Tridiagonal solve", ("tridiagonal_solve",), "MIXED",
     "scipy solve_banded is a forward-elimination/back-substitution recurrence (each row needs the previous one), so it is sequential as run; cyclic reduction or parallel cyclic reduction parallelise it but need a different algorithm and extra arithmetic."),
    ("spline_eval", "Spline evaluation", ("spline_grid_select", "spline_evaluation"), "DP",
     "Each output sample evaluates a cubic on its own interval, and the evaluation-grid selection is a mask plus compaction; independent per sample."),
    ("mean_subtract", "Envelope mean and subtraction", ("envelope_mean_subtract",), "DP",
     "Purely element-wise arithmetic (0.5*(max_env+min_env), imf-mean) and array copies."),
    ("stop_check", "Stopping criterion (IMF check)", ("stop_extrema", "stop_check"), "DP",
     "Reductions (sums of squares, max/min) and element-wise tests plus one extrema detection; the final scalar comparison is sequential but negligible."),
    ("outer_loop", "Outer loop, bookkeeping, end condition", ("outer_loop", "end_condition", "extract_glue"), "SEQ",
     "Sifting iteration k+1 needs the output of iteration k, and IMF j+1 needs the residue of IMF j, so the loop itself is inherently sequential."),
]

# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_n(n: int) -> str:
    """1000 -> '1k', 128000 -> '128k'."""
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def jsonl_append(path: str, records) -> None:
    """Append records to a JSONL file and force them to disk (crash-safe resume)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def jsonl_load(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    LOG.warning("Skipping a truncated line in %s", path)
    return out


def ncores_available() -> int:
    """Cores this process may use (respects affinity where possible).
    PYEMD_PROFILE_NCORES overrides it (manual cap, or a testing hook for exercising 2/4-process code paths)."""
    forced = os.environ.get("PYEMD_PROFILE_NCORES")
    if forced:
        return max(1, int(forced))
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def parse_int_list(text: str):
    return [int(float(t)) for t in str(text).replace(";", ",").split(",") if t.strip()]


def parse_procs(text: str, ncores: int):
    out = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        p = ncores if tok in ("all", "max") else int(tok)
        if 1 <= p <= max(ncores, 1) and p not in out:
            out.append(p)
        elif p > ncores:
            LOG.warning("Skipping processes=%d (> %d available cores)", p, ncores)
    return sorted(out)


# --------------------------------------------------------------------------- #
# System / environment metadata                                               #
# --------------------------------------------------------------------------- #

def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        elif sys.platform.startswith("win"):
            import winreg  # type: ignore
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                      text=True, timeout=5).stdout.strip())
    except Exception:
        pass
    return None


def pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def pyemd_source_info():
    """Versions, source hashes, and the line numbers of every method used in the stage mapping."""
    import PyEMD
    root = os.path.dirname(PyEMD.__file__)
    info = {"version": getattr(PyEMD, "__version__", None) or pkg_version("EMD-signal"),
            "path": root, "sha256": {}, "methods": {}}
    for f in ("EMD.py", "EEMD.py", "CEEMDAN.py", "splines.py", "utils.py"):
        p = os.path.join(root, f)
        if os.path.exists(p):
            info["sha256"][f] = sha256_file(p)
    from PyEMD import EMD, EEMD, CEEMDAN
    wanted = {
        "EMD": ["emd", "find_extrema", "_find_extrema_simple", "_find_extrema_parabol", "prepare_points",
                "prepare_points_simple", "prepare_points_parabol", "extract_max_min_spline", "spline_points",
                "check_imf", "end_condition", "_normalize_time"],
        "EEMD": ["eemd", "_trial_update", "generate_noise", "emd", "ensemble_mean"],
        "CEEMDAN": ["ceemdan", "_decompose_noise", "_eemd", "_trial_update", "generate_noise", "end_condition", "emd"],
    }
    classes = {"EMD": EMD, "EEMD": EEMD, "CEEMDAN": CEEMDAN}
    for cname, names in wanted.items():
        for m in names:
            try:
                fn = inspect.getattr_static(classes[cname], m)
                fn = getattr(fn, "__func__", fn)
                info["methods"][f"{cname}.{m}"] = f"{os.path.basename(inspect.getsourcefile(fn))}:{inspect.getsourcelines(fn)[1]}"
            except Exception:
                info["methods"][f"{cname}.{m}"] = None
    spl = importlib.import_module("PyEMD.splines")
    for m in ("cubic", "cubic_spline_3pts", "akima", "pchip", "cubic_hermite"):
        try:
            info["methods"][f"splines.{m}"] = f"splines.py:{inspect.getsourcelines(getattr(spl, m))[1]}"
        except Exception:
            info["methods"][f"splines.{m}"] = None
    try:
        info["emd_emd_source_sha256"] = sha256_bytes(inspect.getsource(EMD.emd).encode())
    except Exception:
        info["emd_emd_source_sha256"] = None
    return info


def pyemd_defaults():
    """Defaults actually in force (read from freshly built instances, not from docs)."""
    from PyEMD import EMD, EEMD, CEEMDAN
    e = EMD()
    d = {"EMD": {k: getattr(e, k) for k in (
        "spline_kind", "extrema_detection", "nbsym", "std_thr", "svar_thr", "energy_ratio_thr",
        "total_power_thr", "range_thr", "MAX_ITERATION", "FIXE", "FIXE_H", "scale_factor")}}
    d["EMD"]["DTYPE"] = str(getattr(e.DTYPE, "__name__", e.DTYPE))
    d["EMD"]["max_imf (call argument)"] = -1
    ee = EEMD(trials=10, parallel=False)
    d["EEMD"] = {k: getattr(ee, k) for k in ("trials", "noise_width", "noise_kind", "separate_trends", "parallel")}
    d["EEMD"]["signature_defaults"] = {k: str(v.default) for k, v in inspect.signature(EEMD.__init__).parameters.items()
                                      if v.default is not inspect.Parameter.empty}
    ce = CEEMDAN(trials=10, parallel=False)
    d["CEEMDAN"] = {k: getattr(ce, k) for k in ("trials", "epsilon", "noise_scale", "noise_kind", "beta_progress",
                                                "range_thr", "total_power_thr", "_max_imf", "parallel")}
    d["CEEMDAN"]["signature_defaults"] = {k: str(v.default) for k, v in inspect.signature(CEEMDAN.__init__).parameters.items()
                                         if v.default is not inspect.Parameter.empty}
    return d


def collect_metadata(args, thread_mode: str) -> dict:
    import numpy as np
    meta = OrderedDict()
    meta["timestamp_utc"] = now_iso()
    meta["script"] = {"name": os.path.basename(__file__), "version": SCRIPT_VERSION,
                      "sha256": sha256_file(os.path.abspath(__file__)), "argv": sys.argv}
    meta["thread_mode"] = thread_mode
    meta["python"] = {"version": sys.version, "implementation": platform.python_implementation(),
                      "executable": sys.executable}
    meta["packages"] = {n: pkg_version(n) for n in (
        "numpy", "scipy", "pandas", "matplotlib", "EMD-signal", "line_profiler", "psutil", "threadpoolctl", "tqdm")}
    meta["os"] = {"platform": platform.platform(), "system": platform.system(), "release": platform.release(),
                  "machine": platform.machine()}
    meta["cpu"] = {"model": cpu_model(), "logical_cores": os.cpu_count(), "available_cores": ncores_available()}
    try:
        import psutil
        meta["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        meta["cpu"]["freq_mhz_current_max"] = [getattr(freq, "current", None), getattr(freq, "max", None)] if freq else None
    except Exception:
        meta["cpu"]["physical_cores"] = None
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    meta["cpu"]["scaling_governor"] = open(gov).read().strip() if os.path.exists(gov) else None
    meta["ram_bytes"] = ram_bytes()
    try:
        meta["loadavg_at_start"] = list(os.getloadavg())
    except (AttributeError, OSError):
        meta["loadavg_at_start"] = None
    meta["thread_env"] = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    try:
        from threadpoolctl import threadpool_info
        meta["threadpools"] = [{k: v for k, v in d.items() if k != "filepath"} for d in threadpool_info()]
    except Exception:
        meta["threadpools"] = None
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        meta["numpy_blas"] = {k: blas.get(k) for k in ("name", "version", "openblas configuration")}
    except Exception:
        meta["numpy_blas"] = None
    import multiprocessing as mp
    meta["multiprocessing_start_method"] = mp.get_start_method()
    meta["perf_counter_info"] = {k: str(v) for k, v in vars(time.get_clock_info("perf_counter")).items()} \
        if hasattr(time.get_clock_info("perf_counter"), "__dict__") else str(time.get_clock_info("perf_counter"))
    meta["seeds"] = {"signal_seed": args.seed, "noise_seed": args.noise_seed}
    meta["pyemd"] = pyemd_source_info()
    meta["pyemd_defaults"] = pyemd_defaults()
    meta["signals"] = {"fs_hz": FS_HZ, "tones_hz": TONES_HZ, "tone_amplitudes": TONES_AMP, "snr_db_target": SNR_DB,
                       "chirp_am": CHIRP}
    return meta


# --------------------------------------------------------------------------- #
# Test signals (deterministic; fixed seeds)                                   #
# --------------------------------------------------------------------------- #

def make_signal(kind: str, n: int, seed: int):
    """Return a float64 test signal.  Sampling rate FS_HZ, so longer N = longer record."""
    import numpy as np
    t = np.arange(n, dtype=np.float64) / FS_HZ
    if kind in ("multitone", "multitone_noisy"):
        phases = np.random.default_rng(np.random.SeedSequence([seed, 101, n])).uniform(0.0, 2 * np.pi, len(TONES_HZ))
        x = np.zeros(n)
        for a, f, ph in zip(TONES_AMP, TONES_HZ, phases):
            x += a * np.sin(2 * np.pi * f * t + ph)
        if kind == "multitone_noisy":
            sigma = math.sqrt(float(np.mean(x ** 2)) / (10.0 ** (SNR_DB / 10.0)))
            x = x + np.random.default_rng(np.random.SeedSequence([seed, 202, n])).normal(0.0, sigma, n)
    elif kind == "chirp_am":
        dur = n / FS_HZ
        k = (CHIRP["f1"] - CHIRP["f0"]) / dur           # linear sweep f0 -> f1 over the record
        env = 1.0 + CHIRP["am_depth"] * np.sin(2 * np.pi * CHIRP["f_am"] * t)
        x = env * np.sin(2 * np.pi * (CHIRP["f0"] * t + 0.5 * k * t * t))
    else:
        raise ValueError(f"unknown signal kind {kind!r}")
    return np.ascontiguousarray(x, dtype=np.float64)


def measured_snr_db(n: int, seed: int):
    import numpy as np
    clean = make_signal("multitone", n, seed)
    noisy = make_signal("multitone_noisy", n, seed)
    return float(10 * np.log10(np.mean(clean ** 2) / np.mean((noisy - clean) ** 2)))


# --------------------------------------------------------------------------- #
# Stage clock: nested timers with exclusive ("self") and inclusive time       #
# --------------------------------------------------------------------------- #

class StageClock:
    """Nested stage timers.  ``excl[stage]`` is the time spent in the stage itself
    (children subtracted), ``incl[stage]`` includes children.  All times in ns."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.excl = defaultdict(int)
        self.incl = defaultdict(int)
        self.calls = defaultdict(int)
        self.stack = []
        self.override = None          # stage name used instead of the default by wrappers that allow it
        self.events = 0               # number of enter/exit pairs (for the overhead estimate)
        self.emd_calls = 0
        self.emd_calls_by_phase = defaultdict(int)
        self.sift_counts = []         # one list (iterations per IMF) per EMD call
        self.capped = 0               # IMFs that hit MAX_ITERATION
        self.phase = "none"

    def enter(self, stage):
        self.stack.append([stage, perf_counter_ns(), 0])

    def exit(self):
        t1 = perf_counter_ns()
        stage, t0, child = self.stack.pop()
        el = t1 - t0
        self.excl[stage] += el - child
        self.incl[stage] += el
        self.calls[stage] += 1
        self.events += 1
        if self.stack:
            self.stack[-1][2] += el

    def snapshot(self, total_ns, out_rows):
        n_imf_total = sum(len(c) for c in self.sift_counts)
        n_sift = sum(sum(c) for c in self.sift_counts) if self.sift_counts else self.calls.get("extract_glue", 0)
        return {
            "total_ns": int(total_ns),
            "stages": {k: [int(self.excl[k]), int(self.incl[k]), int(self.calls[k])] for k in self.calls},
            "events": int(self.events),
            "n_sift": int(n_sift),
            "n_imf": int(n_imf_total),
            "sift_counts": self.sift_counts if len(self.sift_counts) <= 1 else None,   # per-IMF list only for single EMD
            "capped": int(self.capped),
            "emd_calls": int(self.emd_calls),
            "emd_calls_by_phase": dict(self.emd_calls_by_phase),
            "out_rows": int(out_rows),
        }


def group_of(fine: str) -> str:
    if fine.startswith("phase:"):
        return "ensemble_glue"
    return FINE_TO_GROUP.get(fine, "outer_loop")


def timed(fn, stage, clock, allow_override=False):
    """Wrap ``fn`` so that its (exclusive) time is accumulated under ``stage``."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        clock.enter((clock.override or stage) if allow_override else stage)
        try:
            return fn(*args, **kwargs)
        finally:
            clock.exit()
    return wrapper


def phased(fn, phase, clock):
    """Like ``timed`` but also records the CEEMDAN phase (used to attribute EMD calls)."""
    stage = "phase:" + phase

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        prev = clock.phase
        clock.phase = phase
        clock.enter(stage)
        try:
            return fn(*args, **kwargs)
        finally:
            clock.exit()
            clock.phase = prev
    return wrapper


_MISSING = object()


class Patcher:
    """Remembers every monkeypatch so that all of them are undone (and verified)."""

    def __init__(self):
        self._saved = []
        self.missing = []

    def set(self, owner, name, new, label=None):
        label = label or f"{getattr(owner, '__name__', owner)}.{name}"
        if isinstance(owner, type):
            if name not in owner.__dict__:
                self.missing.append(label)
                return False
            orig = owner.__dict__[name]            # keeps staticmethod/classmethod wrappers intact
        else:
            if not hasattr(owner, name):
                self.missing.append(label)
                return False
            orig = getattr(owner, name)
        self._saved.append((owner, name, orig, new))
        setattr(owner, name, new)
        return True

    def restore(self):
        for owner, name, orig, new in reversed(self._saved):
            setattr(owner, name, orig)
            cur = owner.__dict__.get(name, _MISSING) if isinstance(owner, type) else getattr(owner, name, _MISSING)
            if cur is not orig:
                LOG.error("Failed to restore %s.%s", getattr(owner, "__name__", owner), name)
        self._saved.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


# --------------------------------------------------------------------------- #
# Instrumented copy of EMD.emd (needed for sub-stages that have no method)    #
# --------------------------------------------------------------------------- #

def make_instrumented_emd(clock):
    """Return a copy of PyEMD 1.10.0 ``EMD.emd`` with timers at the sub-stage boundaries.

    The control flow is a line-for-line copy of the library method.  Added lines are
    only the clock calls, the sift counters and the ``stop_extrema`` override.  The
    script verifies that the output is bit-identical to the library's own ``emd``."""
    import numpy as np
    from PyEMD.utils import get_timeline, unify_types

    def emd(self, S, T=None, max_imf=-1):
        clock.emd_calls += 1
        clock.emd_calls_by_phase[clock.phase] += 1
        clock.enter("outer_loop")
        try:
            if T is not None and len(S) != len(T):
                raise ValueError("Time series have different sizes: len(S) -> {} != {} <- len(T)".format(len(S), len(T)))

            if T is None or self.extrema_detection == "simple":
                T = get_timeline(len(S), S.dtype)

            T = self._normalize_time(T)
            S, T = unify_types(S, T)
            self.DTYPE = S.dtype
            N = len(S)

            residue = S.astype(self.DTYPE)
            imf = np.zeros(len(S), dtype=self.DTYPE)
            imf_old = np.nan

            if S.shape != T.shape:
                raise ValueError("Position or time array should be the same size as signal.")

            imfNo = 0
            extNo = -1
            IMF = np.empty((imfNo, N))
            finished = False
            sift_counts = []

            while not finished:
                self.logger.debug("IMF -- %s", imfNo)

                residue[:] = S - np.sum(IMF[:imfNo], axis=0)
                imf = residue.copy()
                mean = np.zeros(len(S), dtype=self.DTYPE)

                n = 0
                n_h = 0
                n_sift = 0

                while True:
                    n += 1
                    if n >= self.MAX_ITERATION:
                        self.logger.info("Max iterations reached for IMF. Continuing with another IMF.")
                        clock.capped += 1
                        break

                    ext_res = self.find_extrema(T, imf)
                    max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]
                    extNo = len(min_pos) + len(max_pos)
                    nzm = len(indzer)

                    if extNo > 2:
                        max_env, min_env, eMax, eMin = self.extract_max_min_spline(T, imf)
                        n_sift += 1

                        clock.enter("envelope_mean_subtract")
                        mean[:] = 0.5 * (max_env + min_env)
                        imf_old = imf.copy()
                        imf[:] = imf - mean
                        clock.exit()

                        if self.FIXE:
                            if n >= self.FIXE:
                                break

                        elif self.FIXE_H:
                            clock.override = "stop_extrema"
                            try:
                                tmp_residue = self.find_extrema(T, imf)
                            finally:
                                clock.override = None
                            max_pos, min_pos, ind_zer = (tmp_residue[0], tmp_residue[2], tmp_residue[4])
                            extNo = len(max_pos) + len(min_pos)
                            nzm = len(ind_zer)

                            if n == 1:
                                continue

                            n_h = n_h + 1 if abs(extNo - nzm) < 2 else 0

                            if n_h >= self.FIXE_H:
                                break

                        else:
                            clock.override = "stop_extrema"
                            try:
                                ext_res = self.find_extrema(T, imf)
                            finally:
                                clock.override = None
                            max_pos, _, min_pos, _, ind_zer = ext_res
                            extNo = len(max_pos) + len(min_pos)
                            nzm = len(ind_zer)

                            if imf_old is np.nan:
                                continue

                            f1 = self.check_imf(imf, imf_old, eMax, eMin)
                            f2 = abs(extNo - nzm) < 2

                            if f1 and f2:
                                break

                    else:  # fewer than 3 extrema: trend
                        finished = True
                        break

                IMF = np.vstack((IMF, imf.copy()))
                imfNo += 1
                sift_counts.append(n_sift)

                if self.end_condition(S, IMF) or imfNo == max_imf:
                    finished = True
                    break

            if extNo <= 2:
                IMF = IMF[:-1]
                sift_counts = sift_counts[:-1]

            self.imfs = IMF.copy()
            self.residue = S - np.sum(self.imfs, axis=0)

            if not np.allclose(self.residue, 0):
                IMF = np.vstack((IMF, self.residue))

            clock.sift_counts.append(sift_counts)
            return IMF
        finally:
            clock.exit()

    return emd


def patch_emd(patcher, clock, use_replica=True):
    """Instrument the EMD class (methods + the module-level spline helpers + scipy's solve_banded)."""
    import numpy as np
    from PyEMD import EMD
    from scipy.interpolate import (Akima1DInterpolator, CubicHermiteSpline, CubicSpline,
                                   PchipInterpolator, interp1d)
    emd_mod = importlib.import_module("PyEMD.EMD")

    for name, stage, ov in (("find_extrema", "extrema_detection", True),
                            ("prepare_points", "boundary_handling", False),
                            ("spline_points", "spline_grid_select", False),
                            ("extract_max_min_spline", "extract_glue", False),
                            ("check_imf", "stop_check", False),
                            ("end_condition", "end_condition", False)):
        orig = EMD.__dict__.get(name)
        if orig is None:
            patcher.missing.append(f"EMD.{name}")
            continue
        patcher.set(EMD, name, timed(orig, stage, clock, ov))

    if use_replica:
        patcher.set(EMD, "emd", make_instrumented_emd(clock))
    else:                                              # fallback: whole emd() body is one "outer_loop" stage
        orig_emd = EMD.__dict__["emd"]

        @functools.wraps(orig_emd)
        def emd_lumped(self, *a, **k):
            clock.emd_calls += 1
            clock.emd_calls_by_phase[clock.phase] += 1
            clock.enter("outer_loop")
            try:
                return orig_emd(self, *a, **k)
            finally:
                clock.exit()
        patcher.set(EMD, "emd", emd_lumped)

    # --- module-level spline helpers: split construction from evaluation ---------------
    def two_phase(ctor):
        def f(X, Y, x):
            clock.enter("spline_construction")
            try:
                spl = ctor(X, Y)
            finally:
                clock.exit()
            clock.enter("spline_evaluation")
            try:
                return spl(x)
            finally:
                clock.exit()
        return f

    def cubic_hermite(X, Y, x):
        clock.enter("spline_construction")
        try:
            dydx = np.gradient(Y, X)
            spl = CubicHermiteSpline(X, Y, dydx)
        finally:
            clock.exit()
        clock.enter("spline_evaluation")
        try:
            return spl(x)
        finally:
            clock.exit()

    class _TimedCallable:
        def __init__(self, obj):
            self._o = obj

        def __call__(self, x):
            clock.enter("spline_evaluation")
            try:
                return self._o(x)
            finally:
                clock.exit()

    def timed_interp1d(*a, **k):
        clock.enter("spline_construction")
        try:
            obj = interp1d(*a, **k)
        finally:
            clock.exit()
        return _TimedCallable(obj)

    repl = {"cubic": two_phase(CubicSpline), "akima": two_phase(Akima1DInterpolator),
            "pchip": two_phase(PchipInterpolator), "cubic_hermite": cubic_hermite,
            "interp1d": timed_interp1d}
    if hasattr(emd_mod, "cubic_spline_3pts"):
        repl["cubic_spline_3pts"] = timed(emd_mod.cubic_spline_3pts, "spline_3pt", clock)
    for name, new in repl.items():
        patcher.set(emd_mod, name, new, label=f"PyEMD.EMD.{name}")

    # --- the tridiagonal solve inside scipy's CubicSpline.__init__ -------------------------
    try:
        cubic_mod = importlib.import_module("scipy.interpolate._cubic")
        patcher.set(cubic_mod, "solve_banded", timed(cubic_mod.solve_banded, "tridiagonal_solve", clock),
                    label="scipy.interpolate._cubic.solve_banded")
    except Exception as exc:  # pragma: no cover - depends on the SciPy version
        patcher.missing.append(f"scipy.interpolate._cubic.solve_banded ({exc})")


def patch_eemd(patcher, clock):
    from PyEMD import EEMD
    for name, stage in (("generate_noise", "noise_generation"), ("_trial_update", "trial_update"),
                        ("emd", "ensemble_glue"), ("ensemble_mean", "ensemble_glue"), ("eemd", "ensemble_glue")):
        orig = EEMD.__dict__.get(name)
        if orig is None:
            patcher.missing.append(f"EEMD.{name}")
            continue
        patcher.set(EEMD, name, timed(orig, stage, clock))


def patch_ceemdan(patcher, clock):
    from PyEMD import CEEMDAN
    d = CEEMDAN.__dict__
    for name, stage in (("generate_noise", "noise_generation"), ("_trial_update", "trial_update"),
                        ("emd", "ensemble_glue")):
        if name in d:
            patcher.set(CEEMDAN, name, timed(d[name], stage, clock))
        else:
            patcher.missing.append(f"CEEMDAN.{name}")
    for name, phase in (("_decompose_noise", "noise_decomposition"), ("_eemd", "first_imf"),
                        ("end_condition", "end_condition"), ("ceemdan", "main_loop")):
        if name in d:
            patcher.set(CEEMDAN, name, phased(d[name], phase, clock))
        else:
            patcher.missing.append(f"CEEMDAN.{name}")


@contextlib.contextmanager
def instrumentation(algo, use_replica=True):
    """Context manager: instrument ``algo`` ('emd'|'eemd'|'ceemdan'); always restores the library."""
    clock = StageClock()
    patcher = Patcher()
    try:
        patch_emd(patcher, clock, use_replica)
        if algo == "eemd":
            patch_eemd(patcher, clock)
        elif algo == "ceemdan":
            patch_ceemdan(patcher, clock)
        yield clock, patcher
    finally:
        patcher.restore()


# --------------------------------------------------------------------------- #
# Runner: one library configuration, callable on a signal                     #
# --------------------------------------------------------------------------- #

class Runner:
    """``procs == 0`` -> serial (parallel=False); ``procs >= 1`` -> parallel=True, processes=procs."""

    def __init__(self, algo, emd_kwargs, max_imf=-1, trials=0, procs=0, noise_seed=12345):
        from PyEMD import EMD, EEMD, CEEMDAN
        self.algo, self.max_imf, self.noise_seed = algo, max_imf, noise_seed
        self.trials, self.procs = trials, procs
        self.emd = EMD(**emd_kwargs)
        if algo == "emd":
            self.obj = self.emd
        else:
            kw = {"trials": trials, "ext_EMD": self.emd, "parallel": procs > 0}
            if procs > 0:
                kw["processes"] = procs
            self.obj = (EEMD if algo == "eemd" else CEEMDAN)(**kw)

    def __call__(self, x):
        if self.algo == "emd":
            return self.emd.emd(x, max_imf=self.max_imf)
        self.obj.noise_seed(self.noise_seed)
        if self.algo == "eemd":
            return self.obj.eemd(x, max_imf=self.max_imf)
        return self.obj.ceemdan(x, max_imf=self.max_imf)

    def noise_settings(self):
        o = self.obj
        if self.algo == "eemd":
            return {"noise_width": o.noise_width, "noise_kind": o.noise_kind, "trials": o.trials,
                    "separate_trends": o.separate_trends, "parallel": o.parallel, "processes": o.processes}
        if self.algo == "ceemdan":
            return {"epsilon": o.epsilon, "noise_scale": o.noise_scale, "noise_kind": o.noise_kind,
                    "beta_progress": o.beta_progress, "range_thr": o.range_thr,
                    "total_power_thr": o.total_power_thr, "trials": o.trials,
                    "parallel": o.parallel, "processes": o.processes}
        return {}


def timed_call(fn, *args):
    """Wall-clock one call (GC collected before, disabled during)."""
    gc.collect()
    gc.disable()
    try:
        t0 = perf_counter_ns()
        out = fn(*args)
        t1 = perf_counter_ns()
    finally:
        gc.enable()
    return t1 - t0, out


def stage_timed_call(runner, x, clock):
    clock.reset()
    t_ns, out = timed_call(runner, x)
    return clock.snapshot(t_ns, out.shape[0]), out


# --------------------------------------------------------------------------- #
# Equality check: instrumentation must not change the numerical output        #
# --------------------------------------------------------------------------- #

def verify_instrumentation(emd_kwargs, max_imf, seed, n=2000):
    """Return (use_replica, stage_ok, details).  Bit-identical output is required."""
    import numpy as np
    details = {"n": n, "cases": []}

    def check(use_replica):
        ok_all = True
        for sig in SIGNALS:
            x = make_signal(sig, n, seed)
            runner = Runner("emd", emd_kwargs, max_imf)
            ref = runner(x)
            with instrumentation("emd", use_replica) as (clock, _p):
                clock.reset()
                inst = Runner("emd", emd_kwargs, max_imf)(x)
            same = ref.shape == inst.shape and bool(np.array_equal(ref, inst))
            details["cases"].append({"signal": sig, "replica": use_replica, "bit_identical": same, "rows": int(ref.shape[0])})
            ok_all &= same
        return ok_all

    if check(True):
        return True, True, details
    LOG.warning("Instrumented emd() copy differs from the library output -> falling back to method-level timers")
    if check(False):
        return False, True, details
    return False, False, details


def verify_ensemble_instrumentation(seed, n=1000):
    import numpy as np
    res = {}
    for algo, trials in (("eemd", 4), ("ceemdan", 3)):
        x = make_signal("multitone_noisy", n, seed)
        ref = Runner(algo, {}, -1, trials, 0)(x)
        with instrumentation(algo, True):
            inst = Runner(algo, {}, -1, trials, 0)(x)
        res[algo] = bool(ref.shape == inst.shape and np.array_equal(ref, inst))
    return res


# --------------------------------------------------------------------------- #
# cProfile, line_profiler, tracemalloc                                        #
# --------------------------------------------------------------------------- #
C_NUMERIC_HINTS = ("numpy", "ndarray", "ufunc", "scipy", "_ppoly", "lapack", "blas", "fortran", "_fitpack")
PY_ITER_BUILTINS = ("builtins.max", "builtins.min", "builtins.sum", "builtins.sorted", "builtins.any", "builtins.all")


def short_path(fn: str) -> str:
    f = fn.replace("\\", "/")
    m = re.search(r"(?:site-packages|dist-packages)/(.*)", f)
    if m:
        return m.group(1)
    return f if f.startswith("<") or f == "~" else os.path.basename(f)


def classify_profile_entry(fn: str, name: str) -> str:
    f = fn.replace("\\", "/")
    if f == "~":
        low = name.lower()
        if any(b in low for b in PY_ITER_BUILTINS):
            return "py_builtin_iter"           # Python-level iteration over the elements when applied to arrays
        if any(h in low for h in C_NUMERIC_HINTS):
            return "compiled_visible"          # NumPy/SciPy C entry points that cProfile sees
        return "py_builtin_other"
    if "/PyEMD/" in f:
        return "py_frame_pyemd"
    if "/numpy/" in f:
        return "py_frame_numpy"
    if "/scipy/" in f:
        return "py_frame_scipy"
    return "py_frame_other"


def run_cprofile(runner, x):
    import cProfile
    import pstats
    pr = cProfile.Profile()
    gc.collect()
    gc.disable()
    try:
        t0 = perf_counter_ns()
        pr.enable()
        out = runner(x)
        pr.disable()
        wall = perf_counter_ns() - t0
    finally:
        gc.enable()
    stats = pstats.Stats(pr).stats
    by_cat = defaultdict(float)
    rows = []
    for (fn, ln, name), (cc, nc, tt, ct, _callers) in stats.items():
        cat = classify_profile_entry(fn, name)
        by_cat[cat] += tt
        rows.append([short_path(fn), int(ln), name, int(nc), int(cc), float(tt), float(ct), cat])
    total_self = float(sum(by_cat.values()))
    py_frames = float(sum(v for k, v in by_cat.items() if k.startswith("py_frame")))
    summary = {"wall_ns": int(wall), "total_self_s": total_self, "py_frames_s": py_frames,
               "by_cat_s": {k: float(v) for k, v in by_cat.items()}, "out_rows": int(out.shape[0])}
    return summary, rows


def run_line_profile(emd_kwargs, max_imf, x):
    """Per-line timings of the hottest PyEMD methods (needs ``line_profiler``)."""
    import linecache
    from line_profiler import LineProfiler
    from PyEMD import EMD
    funcs = []
    for name in ("emd", "extract_max_min_spline", "find_extrema", "_find_extrema_simple", "prepare_points",
                 "prepare_points_simple", "spline_points", "check_imf", "end_condition"):
        fn = inspect.getattr_static(EMD, name, None)
        fn = getattr(fn, "__func__", fn)
        if fn is not None:
            funcs.append(fn)
    spl = importlib.import_module("PyEMD.splines")
    funcs.append(spl.cubic)
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    obj = EMD(**emd_kwargs)
    gc.collect()
    lp.enable_by_count()
    try:
        obj.emd(x, max_imf=max_imf)
    finally:
        lp.disable_by_count()
    stats = lp.get_stats()
    unit = stats.unit
    rows = []
    for (fn, _first, name), lines in stats.timings.items():
        ftotal = sum(t for _l, _h, t in lines) * unit
        for lineno, hits, t in lines:
            sec = t * unit
            rows.append([name, short_path(fn), int(lineno), int(hits), float(sec),
                         float(100.0 * sec / ftotal) if ftotal else 0.0,
                         linecache.getline(fn, lineno).strip()])
    return rows


def run_memory(runner, x):
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    try:
        runner(x)
        cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(cur), int(peak)


# --------------------------------------------------------------------------- #
# Calibrations and diagnostics (small, run once per worker)                   #
# --------------------------------------------------------------------------- #

def calibrate_timer_overhead(n=200_000):
    """Cost of one enter/exit pair and of one wrapped call (used for the overhead estimate)."""
    clk = StageClock()
    t0 = perf_counter_ns()
    for _ in range(n):
        clk.enter("x")
        clk.exit()
    pair_ns = (perf_counter_ns() - t0) / n

    def f():
        return None
    w = timed(f, "w", StageClock())
    t0 = perf_counter_ns()
    for _ in range(n):
        f()
    base = (perf_counter_ns() - t0) / n
    t0 = perf_counter_ns()
    for _ in range(n):
        w()
    wrapped = (perf_counter_ns() - t0) / n
    return {"enter_exit_pair_ns": pair_ns, "wrapped_call_extra_ns": max(wrapped - base, 0.0), "n": n}


def calibrate_floors(emd_kwargs, max_imf, use_replica, seed, reps=200, sizes=None):
    """Per-call fixed cost of every stage, measured on tiny signals where arithmetic is negligible.

    Unit = one call of the wrapped function, except 'outer_loop' whose unit is one sift iteration."""
    import numpy as np
    out = {}
    for n_floor in (sizes or (FLOOR_N, 2 * FLOOR_N)):
        x = make_signal("multitone_noisy", n_floor, seed)
        samples = defaultdict(list)
        with instrumentation("emd", use_replica) as (clock, _p):
            obj = Runner("emd", emd_kwargs, max_imf)
            obj(x)                                             # warm-up
            for _ in range(reps):
                clock.reset()
                obj(x)
                n_sift = sum(sum(c) for c in clock.sift_counts) if clock.sift_counts else clock.calls.get("extract_glue", 0)
                for stage, calls in clock.calls.items():
                    if stage == "outer_loop":
                        if n_sift:
                            samples[stage].append(clock.excl[stage] / n_sift)
                    elif calls:
                        samples[stage].append(clock.excl[stage] / calls)
        out[n_floor] = {st: float(np.median(v)) for st, v in samples.items()}
    return out


class _FakePool:
    """Stand-in for multiprocessing.Pool that runs in-process and records what would be pickled."""
    log = []                      # (what, pickled_bytes, pickle_seconds)
    created = 0

    def __init__(self, processes=None, *a, **k):
        type(self).created += 1

    def _rec(self, func, kind):
        t0 = perf_counter_ns()
        size = len(pickle.dumps(func))
        type(self).log.append((kind, size, (perf_counter_ns() - t0) / 1e9))

    def map(self, func, iterable, *a, **k):
        self._rec(func, "map")
        return [func(i) for i in iterable]

    def imap_unordered(self, func, iterable, *a, **k):
        self._rec(func, "imap_unordered")
        return (func(i) for i in iterable)

    def close(self):
        pass

    def join(self):
        pass


def diag_redundancy(seed, n=2000):
    """How many find_extrema() calls receive an array identical to an earlier call's?"""
    import numpy as np
    from PyEMD import EMD
    res = {}
    for sig in SIGNALS:
        x = make_signal(sig, n, seed)
        seen = []
        orig = EMD.__dict__["find_extrema"]

        def spy(self, T, S, _o=orig):
            seen.append(zlib.crc32(np.ascontiguousarray(S).view(np.uint8)))
            return _o(self, T, S)
        counts = defaultdict(int)
        orig_ex = EMD.__dict__["extract_max_min_spline"]

        def spy_ex(self, T, S, _o=orig_ex):
            counts["sift"] += 1
            return _o(self, T, S)
        with Patcher() as p:
            p.set(EMD, "find_extrema", spy)
            p.set(EMD, "extract_max_min_spline", spy_ex)
            EMD().emd(x)
        total = len(seen)
        distinct = len(set(seen))
        res[sig] = {"n": n, "find_extrema_calls": total, "distinct_inputs": distinct,
                    "sift_iterations": counts["sift"],
                    "calls_per_sift_iteration": total / counts["sift"] if counts["sift"] else None,
                    "redundant_fraction": 1.0 - distinct / total if total else None}
    return res


def diag_parallel_semantics(seed, n, trials=8, procs=2):
    """Verified facts about the parallel code paths (see README): duplicated EEMD noise,
    number of Pools CEEMDAN creates, and the bytes pickled per Pool task."""
    import numpy as np
    from PyEMD import EEMD, CEEMDAN
    x = make_signal("multitone_noisy", n, seed)
    out = {"n": n, "trials": trials, "procs": procs}

    def distinct_rows(a):
        return len({np.ascontiguousarray(r).tobytes() for r in a})
    try:
        e_par = EEMD(trials=trials, parallel=True, processes=procs)
        e_par.noise_seed(seed)
        e_par.eemd(x)
        e_ser = EEMD(trials=trials, parallel=False)
        e_ser.noise_seed(seed)
        e_ser.eemd(x)
        out["eemd_distinct_first_imf_parallel"] = distinct_rows(e_par.all_imfs[0])
        out["eemd_distinct_first_imf_serial"] = distinct_rows(e_ser.all_imfs[0])
    except Exception as exc:
        out["eemd_noise_check_error"] = repr(exc)
    for algo, modname, cls in (("eemd", "PyEMD.EEMD", EEMD), ("ceemdan", "PyEMD.CEEMDAN", CEEMDAN)):
        try:
            mod = sys.modules[modname]
            _FakePool.log, _FakePool.created = [], 0
            with Patcher() as p:
                p.set(mod, "Pool", _FakePool, label=f"{modname}.Pool")
                obj = cls(trials=trials, parallel=True, processes=procs)
                obj.noise_seed(seed)
                comps = obj.eemd(x) if algo == "eemd" else obj.ceemdan(x)
            out[f"{algo}_pools_created"] = _FakePool.created
            out[f"{algo}_pool_calls"] = [{"call": k, "pickled_bytes": s, "pickle_seconds": t}
                                         for k, s, t in _FakePool.log]
            out[f"{algo}_components"] = int(comps.shape[0])
        except Exception as exc:
            out[f"{algo}_pool_check_error"] = repr(exc)
    return out


def _noop(i):
    return i


def _import_task(i):
    """What a real EEMD/CEEMDAN worker must do before it can run its first task: import PyEMD (NumPy, SciPy, ...)."""
    import PyEMD  # noqa: F401
    return i


def diag_pool_startup(procs_list, repeats=5):
    """Seconds to create + use + close a multiprocessing.Pool (median of ``repeats``).
    Key ``str(p)`` = workers import PyEMD first (realistic: this is what each EEMD/CEEMDAN call pays; large with
    'spawn' on Windows/macOS); key ``empty_<p>`` = trivial task only."""
    import multiprocessing as mp
    import numpy as np
    out = {}
    for p in procs_list:
        for prefix, fn in (("", _import_task), ("empty_", _noop)):
            ts = []
            for _ in range(repeats):
                t0 = perf_counter_ns()
                pool = mp.Pool(processes=p)
                pool.map(fn, range(p))
                pool.close()
                pool.join()
                ts.append(perf_counter_ns() - t0)
            out[prefix + str(p)] = float(np.median(ts)) / 1e9
    return out


# --------------------------------------------------------------------------- #
# Configuration, plan of measurement units                                     #
# --------------------------------------------------------------------------- #
MODE_DEFAULTS = {
    "quick": dict(lengths=[1000, 2000, 4000], warmup=1, reps=5, reps_ensemble=3, trials=[10, 20], procs="1,2",
                  ens_lengths=[1000, 2000], ref_length=1000, thread_modes="pinned", cprofile_reps=1,
                  line_profile_lengths=[2000]),
    "full": dict(lengths=FULL_LENGTHS, warmup=2, reps=10, reps_ensemble=5, trials=[10, 50, 100], procs="1,2,4,all",
                 ens_lengths=[1000, 2000, 4000, 8000], ref_length=4000, thread_modes="pinned,default",
                 cprofile_reps=3, line_profile_lengths=[1000, 16000, 128000]),
}
MEASURE_ORDER = ["timing", "stage", "cprofile", "line", "memory"]


def _load_json_arg(text):
    if text is None:
        return None
    text = text.strip()
    try:
        if text.startswith("@"):
            with open(text[1:], "r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(text)
    except FileNotFoundError:
        raise SystemExit(f"--custom-emd: file {text[1:]!r} not found.  Create it (see README section 3) or pass the JSON inline.")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--custom-emd: invalid JSON ({exc}).  Tip: use --custom-emd @custom_emd.json to avoid shell quoting problems.")


def resolve_config(a: argparse.Namespace) -> argparse.Namespace:
    """Fill mode defaults and derive everything the plan needs (pure function of the CLI)."""
    d = MODE_DEFAULTS[a.mode]
    c = argparse.Namespace(**vars(a))
    c.lengths = parse_int_list(a.lengths) if a.lengths else list(d["lengths"])
    if a.max_length:
        c.lengths = [n for n in c.lengths if n <= a.max_length] or [min(c.lengths)]
    c.signals = [s.strip() for s in a.signals.split(",")] if a.signals else list(SIGNALS)
    bad = [s for s in c.signals if s not in SIGNALS]
    if bad:
        raise SystemExit(f"Unknown signal(s) {bad}; choose from {SIGNALS}")
    c.warmup = d["warmup"] if a.warmup is None else a.warmup
    c.reps = d["reps"] if a.reps is None else a.reps
    c.reps_ensemble = d["reps_ensemble"] if a.reps_ensemble is None else a.reps_ensemble
    c.stage_reps = c.reps if a.stage_reps is None else a.stage_reps
    c.cprofile_reps = d["cprofile_reps"] if a.cprofile_reps is None else a.cprofile_reps
    c.trials = parse_int_list(a.trials) if a.trials else list(d["trials"])
    c.ncores = ncores_available()
    c.procs_list = parse_procs(a.procs or d["procs"], c.ncores)
    c.ens_lengths = parse_int_list(a.ensemble_lengths) if a.ensemble_lengths else list(d["ens_lengths"])
    if a.max_length:
        c.ens_lengths = [n for n in c.ens_lengths if n <= a.max_length] or [min(c.lengths)]
    c.ref_length = a.ref_length or d["ref_length"]
    if a.max_length and c.ref_length > a.max_length:
        c.ref_length = max(c.ens_lengths)
    c.ref_signal = a.ref_signal
    c.thread_modes = [t.strip() for t in (a.thread_modes or d["thread_modes"]).split(",") if t.strip()]
    for t in c.thread_modes:
        if t not in ("pinned", "default"):
            raise SystemExit("--thread-modes must be a comma list of: pinned, default")
    c.line_lengths = parse_int_list(a.line_profile_lengths) if a.line_profile_lengths else list(d["line_profile_lengths"])
    c.line_lengths = [n for n in c.line_lengths if n in c.lengths] or [min(c.lengths)]
    # EMD configurations: "default" always, "custom" only when the user supplies parameters
    c.emd_configs_map = OrderedDict()
    c.emd_configs_map["default"] = ({}, -1)
    custom = _load_json_arg(a.custom_emd)
    if custom is not None:
        if not isinstance(custom, dict):
            raise SystemExit("--custom-emd must be a JSON object")
        if "DTYPE" in custom:
            import numpy as np
            custom["DTYPE"] = np.dtype(custom["DTYPE"]).type
        c.emd_configs_map["custom"] = (custom, int(a.custom_max_imf))
    c.custom_emd_json = custom
    return c


def build_plan(c: argparse.Namespace):
    """Ordered list of measurement units; every unit is one (algo, config, signal, N, trials, procs)."""
    units = OrderedDict()

    def add(tm, algo, config, signal, n, trials, procs, measures, reps, block):
        key = f"{tm}|{algo}|{config}|{signal}|{n}|{trials}|{procs}"
        if key in units:
            u = units[key]
            u["measures"] = [m for m in MEASURE_ORDER if m in set(u["measures"]) | set(measures)]
            u["blocks"].append(block)
            return
        is_ens = algo != "emd"
        units[key] = {"unit": key, "thread_mode": tm, "algo": algo, "config": config, "signal": signal, "n": n,
                      "trials": trials, "procs": procs, "reps": reps,
                      "stage_reps": c.reps_ensemble if is_ens else c.stage_reps,
                      "cprofile_reps": 1 if is_ens else c.cprofile_reps,
                      "measures": [m for m in MEASURE_ORDER if m in measures], "blocks": [block]}

    for tm in c.thread_modes:
        pinned = tm == "pinned"
        if not c.only or c.only == "emd":
            for config in c.emd_configs_map:
                for signal in c.signals:
                    for n in c.lengths:
                        m = ["timing"]
                        if pinned:
                            if not c.no_stage:
                                m.append("stage")
                            if not c.no_cprofile:
                                m.append("cprofile")
                            if not c.no_memory:
                                m.append("memory")
                            if not c.no_line_profile and n in c.line_lengths:
                                m.append("line")
                        add(tm, "emd", config, signal, n, 0, 0, m, c.reps, "emd")
        if not c.only or c.only == "ensembles":
            t0 = min(c.trials)
            for algo in ("eemd", "ceemdan"):
                for signal in c.signals:                       # block A: per-trial EMD cost vs N and signal
                    for n in c.ens_lengths:
                        m = ["timing"]
                        if pinned:
                            m += [x for x, off in (("stage", c.no_stage), ("cprofile", c.no_cprofile),
                                                   ("memory", c.no_memory)) if not off]
                        add(tm, algo, "default", signal, n, t0, 0, m, c.reps_ensemble, "ens_A")
                if pinned:                                     # blocks B/C at the reference signal and length
                    for trials in c.trials:
                        add(tm, algo, "default", c.ref_signal, c.ref_length, trials, 0, ["timing"], c.reps_ensemble, "ens_B")
                        for p in c.procs_list:
                            add(tm, algo, "default", c.ref_signal, c.ref_length, trials, p, ["timing"], c.reps_ensemble, "ens_C")
    return list(units.values())


def describe_plan(plan, c):
    by_mode = defaultdict(lambda: defaultdict(int))
    for u in plan:
        by_mode[u["thread_mode"]][u["algo"] + (" parallel" if u["procs"] else "")] += 1
    lines = [f"Plan: {len(plan)} measurement units"]
    for tm, d in by_mode.items():
        lines.append(f"  threads={tm}: " + ", ".join(f"{k}={v}" for k, v in d.items()))
    lines.append(f"  lengths={c.lengths}  signals={c.signals}")
    lines.append(f"  EMD configs={list(c.emd_configs_map)}  warmup={c.warmup}  reps(EMD)={c.reps}  reps(ensembles)={c.reps_ensemble}")
    lines.append(f"  ensemble lengths={c.ens_lengths}  trials={c.trials}  procs={c.procs_list}  "
                 f"ref=({c.ref_signal}, N={c.ref_length})  cores available={c.ncores}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Runtime estimator: one real EMD run per planned (signal, N) cell             #
# --------------------------------------------------------------------------- #

def estimate_plan(plan, c, quiet=False, cached=None):
    """Time one EMD run per (signal, N) cell and combine with a simple cost model.
    Strongly data-dependent: PyEMD's runtime on noise-free signals can jump with N, so cells are measured, not extrapolated.
    ``cached`` (the contents of a previous estimate.json) skips the measurements and re-applies the cost model."""
    cells = sorted({(u["signal"], u["n"]) for u in plan}, key=lambda t: (t[1], t[0]))
    t_wall0 = time.time()
    if cached:
        t_emd = {tuple([k.split("|")[0], int(k.split("|")[1])]): v for k, v in cached["t_emd"].items()}
        kc, pool_s = cached["ceemdan_cost_factor"], cached.get("pool_startup_s") or {}
    else:
        logging.getLogger("PyEMD").setLevel(logging.ERROR)
        runner = Runner("emd", {}, -1)
        runner(make_signal("multitone_noisy", 1000, c.seed))                         # import/warm-up
        t_emd = {}
        for signal, n in cells:
            x = make_signal(signal, n, c.seed)
            t_emd[(signal, n)] = timed_call(runner, x)[0] / 1e9
            if not quiet:
                print(f"  estimate: EMD {signal:16s} N={n:7d}: {t_emd[(signal, n)]:8.3f} s", flush=True)
        x = make_signal("multitone_noisy", 1000, c.seed)
        tc = timed_call(Runner("ceemdan", {}, -1, 5, 0, c.noise_seed), x)[0] / 1e9
        kc = tc / (5 * t_emd.get(("multitone_noisy", 1000), t_emd[min(t_emd)]))      # CEEMDAN cost per trial relative to one EMD
        kc = max(kc, 1.0)
        pool_s = diag_pool_startup(c.procs_list, repeats=2) if any(u["procs"] for u in plan) else {}
    total = defaultdict(float)
    for u in plan:
        t1 = t_emd[(u["signal"], u["n"])]
        if u["algo"] == "emd":
            t = t1
        elif u["algo"] == "eemd":
            t = u["trials"] * t1 * 1.03
        else:
            t = u["trials"] * t1 * kc
        if u["procs"]:
            p = min(u["procs"], c.ncores)
            fp = 1.0 if u["algo"] == "eemd" else 0.5          # CEEMDAN: only part of the work uses the Pool (assumed 50 %)
            t = t * ((1 - fp) + fp / p) + (1 if u["algo"] == "eemd" else 2) * pool_s.get(str(u["procs"]), 0.1)
        slow = t > c.slow_run_seconds
        reps = min(u["reps"], c.reps_slow) if slow else u["reps"]
        runs = 0.0
        for m in u["measures"]:
            if m == "timing":
                runs += c.warmup + reps
            elif m == "stage":
                runs += (min(u["stage_reps"], 3) if slow else u["stage_reps"]) * 1.1
            elif m == "cprofile":
                runs += (1 if slow else u["cprofile_reps"]) * 1.5
            elif m == "line":
                runs += 1.8
            elif m == "memory":
                runs += 2.5
        total[f"{u['thread_mode']} / {u['algo']}" + (" parallel" if u["procs"] else "")] += t * runs
    grand = sum(total.values())
    return {"seconds_total": grand, "by_block": dict(total), "t_emd": {f"{k[0]}|{k[1]}": v for k, v in t_emd.items()},
            "ceemdan_cost_factor": kc, "calibration_wall_s": time.time() - t_wall0, "pool_startup_s": pool_s}


def fmt_duration(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def print_estimate(est):
    print("\nEstimated wall-clock time on THIS machine (measured cell times x run counts; +-30 %):")
    for k, v in sorted(est["by_block"].items()):
        print(f"  {k:36s} {fmt_duration(v):>10s}")
    print(f"  {'TOTAL':36s} {fmt_duration(est['seconds_total']):>10s}")
    print("  (CEEMDAN parallel assumes 50 % of the work uses the Pool; per-cell EMD times above are single measured runs.)\n")


# --------------------------------------------------------------------------- #
# Measurement worker (runs in its own process with controlled thread env)      #
# --------------------------------------------------------------------------- #

class WorkerContext:
    def __init__(self, c, thread_mode):
        self.c = c
        self.thread_mode = thread_mode
        self.seed, self.noise_seed = c.seed, c.noise_seed
        self.warmup = c.warmup
        self.slow_seconds, self.reps_slow = c.slow_run_seconds, c.reps_slow
        self.floor_reps = 60                     # tiny-signal repetitions per floor snapshot (taken before and after each unit)
        self._states = {}
        self.line_ok = not c.no_line_profile
        if self.line_ok:
            try:
                import line_profiler  # noqa: F401  (availability check)
            except Exception:
                LOG.warning("line_profiler not installed -> line-level profiling skipped")
                self.line_ok = False
        self.records_pending = []

    def config_state(self, name):
        if name not in self._states:
            kwargs, max_imf = self.c.emd_configs_map[name]
            use_replica, stage_ok, details = True, True, {}
            if not self.c.no_stage:
                use_replica, stage_ok, details = verify_instrumentation(kwargs, max_imf, self.seed)
                LOG.info("config %-8s instrumented emd() copy bit-identical: %s (stage timers usable: %s)",
                         name, use_replica, stage_ok)
            self._states[name] = {"kwargs": kwargs, "max_imf": max_imf, "use_replica": use_replica,
                                  "stage_ok": stage_ok, "details": details}
        return self._states[name]


def run_canary(ctx, when):
    """A fixed small EMD workload timed repeatedly during the run: quantifies machine drift (thermal, neighbours, ...)."""
    import numpy as np
    x = make_signal("multitone_noisy", 4000, ctx.seed)
    r = Runner("emd", {}, -1)
    timed_call(r, x)
    ts = [timed_call(r, x)[0] for _ in range(5)]
    return {"rec": "canary", "when": when, "thread_mode": ctx.thread_mode, "timestamp_utc": now_iso(),
            "median_ns": int(np.median(ts)), "min_ns": int(min(ts)), "reps": len(ts)}


def run_unit(u, ctx):
    """Execute all measurements of one unit; returns the list of records (written atomically by the caller)."""
    import numpy as np
    base = {k: u[k] for k in ("unit", "thread_mode", "algo", "config", "signal", "n", "trials", "procs")}
    st = ctx.config_state(u["config"])
    x = make_signal(u["signal"], u["n"], ctx.seed)
    runner = Runner(u["algo"], st["kwargs"], st["max_imf"], u["trials"], u["procs"], ctx.noise_seed)
    measures = set(u["measures"])
    recs = []
    t_unit0 = time.time()
    settings = {"noise": runner.noise_settings(), "signal_rms": float(np.sqrt(np.mean(x ** 2))), "pyemd_emd_kwargs": {k: str(v) for k, v in st["kwargs"].items()},
                "max_imf": st["max_imf"]}
    if u["signal"] == "multitone_noisy":
        settings["snr_db_measured"] = measured_snr_db(u["n"], ctx.seed)
    recs.append({"rec": "settings", "data": settings, **base})

    warm = []
    for w in range(ctx.warmup):
        t_ns, out = timed_call(runner, x)
        warm.append(t_ns)
        recs.append({"rec": "timing", "warmup": True, "rep": -(w + 1), "t_ns": int(t_ns), "out_rows": int(out.shape[0]), **base})
    slow = bool(warm) and (float(np.median(warm)) / 1e9) > ctx.slow_seconds
    reps = min(u["reps"], ctx.reps_slow) if slow else u["reps"]
    do_stage = "stage" in measures and st["stage_ok"]
    n_stage = (min(u["stage_reps"], 3) if slow else u["stage_reps"]) if do_stage else 0
    floors_before = None
    if do_stage:                                         # one discarded instrumented run (first-call effects)
        with instrumentation(u["algo"], st["use_replica"]) as (clock, _p):
            stage_timed_call(runner, x, clock)
        if u["algo"] == "emd":                           # per-call floors in the same machine state as this unit's measurements
            floors_before = calibrate_floors(st["kwargs"], st["max_imf"], st["use_replica"], ctx.seed,
                                             reps=ctx.floor_reps, sizes=(FLOOR_N,))[FLOOR_N]
    # plain and stage-timed repetitions are interleaved so that slow drift of the machine cancels in their comparison
    for r in range(max(reps, n_stage)):
        if r < reps:
            t_ns, out = timed_call(runner, x)
            recs.append({"rec": "timing", "warmup": False, "rep": r, "t_ns": int(t_ns), "out_rows": int(out.shape[0]), **base})
        if r < n_stage:
            with instrumentation(u["algo"], st["use_replica"]) as (clock, patcher):
                snap, _out = stage_timed_call(runner, x, clock)
                missing = list(patcher.missing)
            recs.append({"rec": "stage", "rep": r, "replica": bool(st["use_replica"]), "patch_missing": missing, **snap, **base})
    if floors_before is not None:
        floors_after = calibrate_floors(st["kwargs"], st["max_imf"], st["use_replica"], ctx.seed,
                                        reps=ctx.floor_reps, sizes=(FLOOR_N,))[FLOOR_N]
        recs.append({"rec": "floor", "scope": "unit", "data": {"before": floors_before, "after": floors_after}, **base})
    if "cprofile" in measures:
        for r in range(1 if slow else u["cprofile_reps"]):
            summ, rows = run_cprofile(runner, x)
            recs.append({"rec": "cprofile_run", "rep": r, **summ, **base})
            if r == 0:
                recs.append({"rec": "cprofile_fn", "rep": 0, "rows": rows, **base})
    if "line" in measures and ctx.line_ok and u["algo"] == "emd":
        try:
            rows = run_line_profile(st["kwargs"], st["max_imf"], x)
            recs.append({"rec": "line", "rows": rows, **base})
        except Exception as exc:                                    # line_profiler is optional
            LOG.warning("line_profiler failed (%r) -> skipped from now on", exc)
            ctx.line_ok = False
    if "memory" in measures:
        cur, peak = run_memory(runner, x)
        recs.append({"rec": "memory", "current_bytes": cur, "peak_bytes": peak, **base})
    recs.append({"rec": "unit_done", "ok": True, "seconds": time.time() - t_unit0, "reps_done": reps,
                 "reps_planned": u["reps"], "slow_run_rule_applied": slow, **base})
    return recs


def worker_main(c):
    tm = c.worker_thread_mode
    rawdir = os.path.join(c.outdir, "raw", tm)
    os.makedirs(rawdir, exist_ok=True)
    rec_path = os.path.join(rawdir, "records.jsonl")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{tm}] %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(rawdir, "worker.log"), encoding="utf-8")])
    logging.getLogger("PyEMD").setLevel(logging.ERROR)
    import numpy as np  # noqa: F401  (imported only now, after the thread environment was set by the parent)
    limiter = None
    if tm == "pinned":
        try:
            from threadpoolctl import threadpool_limits
            limiter = threadpool_limits(limits=1)
        except Exception:
            LOG.warning("threadpoolctl unavailable; relying on the *_NUM_THREADS environment variables only")
    plan = [u for u in build_plan(c) if u["thread_mode"] == tm]
    existing = jsonl_load(rec_path)
    done = {r["unit"] for r in existing if r.get("rec") == "unit_done"} if c.resume else set()
    have_diag = {r["name"] for r in existing if r.get("rec") == "diag"} if c.resume else set()
    jsonl_append(rec_path, [{"rec": "meta", "thread_mode": tm, "data": collect_metadata(c, tm)}])
    ctx = WorkerContext(c, tm)
    LOG.info("worker start: %d units (%d already done); BLAS/OpenMP threads: %s", len(plan), len(done),
             {k: os.environ.get(k) for k in THREAD_ENV_VARS[:3]})

    # ---- one-time calibrations and diagnostics -------------------------------------------
    def diag(name, fn):
        if name in have_diag:
            return
        try:
            data = fn()
        except Exception as exc:
            LOG.warning("diagnostic %s failed: %r", name, exc)
            data = {"error": repr(exc)}
        jsonl_append(rec_path, [{"rec": "diag", "name": name, "thread_mode": tm, "data": data}])

    if tm == "pinned":
        diag("timer_overhead", calibrate_timer_overhead)
        diag("redundant_find_extrema", lambda: diag_redundancy(c.seed))
        if any(u["algo"] != "emd" for u in plan):
            diag("ensemble_instrumentation_equal", lambda: verify_ensemble_instrumentation(c.seed))
            diag("parallel_semantics", lambda: diag_parallel_semantics(c.seed, c.ref_length if c.ref_length <= 2000 else 1000))
        if c.procs_list and any(u["procs"] for u in plan):
            diag("pool_startup_s", lambda: diag_pool_startup(c.procs_list))
        if not c.no_stage:
            for name in c.emd_configs_map:
                stt = ctx.config_state(name)
                jsonl_append(rec_path, [{"rec": "diag", "name": f"instrumentation_equality[{name}]", "thread_mode": tm,
                                         "data": {"replica_bit_identical": stt["use_replica"], "stage_timers_usable": stt["stage_ok"],
                                                  "cases": stt["details"].get("cases")}}])
                if f"floor[{name}]" not in have_diag and stt["stage_ok"]:
                    fl = calibrate_floors(stt["kwargs"], stt["max_imf"], stt["use_replica"], c.seed)
                    jsonl_append(rec_path, [{"rec": "floor", "name": f"floor[{name}]", "config": name, "thread_mode": tm,
                                             "data": {str(k): v for k, v in fl.items()}}])
                    have_diag.add(f"floor[{name}]")

    # ---- the units ----------------------------------------------------------------------------
    weights = [max(1, u["trials"]) * (u["n"] ** 1.2) * (1 + len(u["measures"])) for u in plan]
    wtotal, wdone, t_start = sum(weights) or 1.0, 0.0, time.time()
    jsonl_append(rec_path, [run_canary(ctx, "start")])
    n_run = 0
    for i, (u, w) in enumerate(zip(plan, weights), 1):
        if u["unit"] in done:
            wdone += w
            continue
        if n_run and n_run % 10 == 0:
            jsonl_append(rec_path, [run_canary(ctx, "during")])
        n_run += 1
        t0 = time.time()
        try:
            recs = run_unit(u, ctx)
        except Exception as exc:
            LOG.error("unit %s FAILED: %r\n%s", u["unit"], exc, traceback.format_exc())
            recs = [{"rec": "error", "error": repr(exc), "traceback": traceback.format_exc(), **{k: u[k] for k in ("unit", "thread_mode", "algo", "config", "signal", "n", "trials", "procs")}},
                    {"rec": "unit_done", "ok": False, "seconds": time.time() - t0, **{k: u[k] for k in ("unit", "thread_mode", "algo", "config", "signal", "n", "trials", "procs")}}]
        jsonl_append(rec_path, recs)
        wdone += w
        el = time.time() - t_start
        eta = el * (wtotal - wdone) / max(wdone, 1e-9)
        LOG.info("[%d/%d] %-52s %7.1fs | elapsed %s | rough ETA %s", i, len(plan), u["unit"], time.time() - t0,
                 fmt_duration(el), fmt_duration(eta))
    jsonl_append(rec_path, [run_canary(ctx, "end"), {"rec": "worker_done", "thread_mode": tm, "timestamp_utc": now_iso()}])
    if limiter is not None:
        limiter.restore_original_limits()
    LOG.info("worker finished")
    return 0


# =========================================================================== #
#                                 REPORTING                                   #
# =========================================================================== #
import csv

CELL_KEYS = ("thread_mode", "algo", "config", "signal", "n", "trials", "procs")
ALGO_ORDER = {"emd": 0, "eemd": 1, "ceemdan": 2}
TM_ORDER = {"pinned": 0, "default": 1}
CPROF_CATS = ["compiled_visible", "py_builtin_iter", "py_builtin_other",
              "py_frame_pyemd", "py_frame_numpy", "py_frame_scipy", "py_frame_other"]
CPROF_LABEL = {
    "compiled_visible": "NumPy/SciPy C entry points seen by cProfile",
    "py_builtin_iter": "Python builtins iterating over elements (max/min/sum...)",
    "py_builtin_other": "other Python builtins",
    "py_frame_pyemd": "Python frames: PyEMD (self time)",
    "py_frame_numpy": "Python frames: NumPy wrappers (self time)",
    "py_frame_scipy": "Python frames: SciPy (self time)",
    "py_frame_other": "Python frames: other (self time)",
}


def ckey(r):
    return tuple(r[k] for k in CELL_KEYS)


def sort_key(k):
    tm, algo, config, signal, n, trials, procs = k
    return (TM_ORDER.get(tm, 9), ALGO_ORDER.get(algo, 9), config != "default", config,
            SIGNALS.index(signal) if signal in SIGNALS else 9, trials, procs, n)


def qstats(a):
    import numpy as np
    a = np.asarray(list(a), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(a.min()), "max": float(a.max())}


def med(a):
    s = qstats(a)
    return s["median"] if s else float("nan")


class Data:
    pass


def load_data(outdir):
    d = Data()
    rawdir = os.path.join(outdir, "raw")
    recs = []
    for tm in sorted(os.listdir(rawdir)) if os.path.isdir(rawdir) else []:
        recs += jsonl_load(os.path.join(rawdir, tm, "records.jsonl"))
    d.recs = recs
    seen, timing = set(), []
    for r in recs:
        if r.get("rec") == "timing":
            k = (r["unit"], r["rep"], r["warmup"])
            if k not in seen:                                  # tolerate accidental duplicates
                seen.add(k)
                timing.append(r)
    d.timing = timing
    pick = lambda name: [r for r in recs if r.get("rec") == name]  # noqa: E731
    d.stage, d.cprof_runs, d.cprof_fn = pick("stage"), pick("cprofile_run"), pick("cprofile_fn")
    d.lines, d.memory, d.settings = pick("line"), pick("memory"), pick("settings")
    d.unit_done, d.errors = pick("unit_done"), pick("error")
    d.diag = {r["name"]: r["data"] for r in pick("diag")}
    d.floor = {r["config"]: r["data"] for r in pick("floor") if r.get("scope") != "unit"}
    d.floor_unit = defaultdict(list)
    for r in pick("floor"):
        if r.get("scope") == "unit":
            d.floor_unit[ckey(r)].append(r)
    d.meta = {r["thread_mode"]: r["data"] for r in pick("meta")}
    d.canary = pick("canary")
    return d


def write_csv(path, rows, columns=None):
    """Write dict rows; the header is the union of all keys (first-appearance order), so rows of different shape lose nothing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    if columns is None:
        columns, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


# ----- LaTeX ------------------------------------------------------------------------ #
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


class Raw(str):
    """A string that is already LaTeX."""


def tex(s):
    if isinstance(s, Raw):
        return str(s)
    return re.sub("|".join(re.escape(k) for k in _TEX), lambda m: _TEX[m.group(0)], str(s))


def write_latex(path, header, rows, align=None, note=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    align = align or ("l" + "r" * (len(header) - 1))
    lines = ["% Generated by pyemd_profile.py -- requires \\usepackage{booktabs}"]
    if note:
        lines.append("% " + note.replace("\n", " "))
    lines += ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(str(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(tex(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def f1(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.{nd}f}"


def f_ms(sec):
    return "--" if sec is None or (isinstance(sec, float) and math.isnan(sec)) else (f"{sec * 1e3:.2f}" if sec < 1 else f"{sec * 1e3:.0f}")


# ----- stage helpers ----------------------------------------------------------------- #

def group_times(stages):
    g = defaultdict(int)
    for fine, v in stages.items():
        g[group_of(fine)] += v[0]
    return g


def stage_cells(d, algo=None):
    cells = defaultdict(list)
    for r in d.stage:
        if algo and r["algo"] != algo:
            continue
        cells[ckey(r)].append(r)
    return dict(sorted(cells.items(), key=lambda kv: sort_key(kv[0])))


def cell_dict(k):
    return dict(zip(CELL_KEYS, k))


# ----- 1. runtime ---------------------------------------------------------------------- #

def agg_runtime(d):
    import numpy as np
    vals, rows_out, slow = defaultdict(list), defaultdict(list), {}
    for r in d.timing:
        if not r["warmup"]:
            vals[ckey(r)].append(r["t_ns"] / 1e9)
            rows_out[ckey(r)].append(r["out_rows"])
    for u in d.unit_done:
        slow[ckey(u)] = (u.get("reps_done"), u.get("reps_planned"), u.get("slow_run_rule_applied"))
    out = []
    for k in sorted(vals, key=sort_key):
        s = qstats(vals[k])
        reps_done, reps_planned, slow_rule = slow.get(k, (None, None, None))
        out.append({**cell_dict(k), "n_reps": s["n"], "median_s": s["median"], "q1_s": s["q1"], "q3_s": s["q3"],
                    "iqr_s": s["iqr"], "min_s": s["min"], "max_s": s["max"], "out_rows": int(np.median(rows_out[k])),
                    "reps_planned": reps_planned, "slow_run_rule_applied": slow_rule})
    return out


# ----- 2. stage fractions (stage timers = primary source) --------------------------------- #

def agg_stage_groups(d):
    out, fine_out = [], []
    for k, runs in stage_cells(d).items():
        c = cell_dict(k)
        per_pct, per_ms, tri_pct = defaultdict(list), defaultdict(list), []
        for r in runs:
            gt = group_times(r["stages"])
            tot = sum(gt.get(g, 0) for g in EMD_GROUPS)
            if tot <= 0:
                continue
            for g in EMD_GROUPS:
                per_pct[g].append(100.0 * gt.get(g, 0) / tot)
                per_ms[g].append(gt.get(g, 0) / 1e6)
            tri_pct.append(100.0 * r["stages"].get("tridiagonal_solve", [0])[0] / tot)
        if not per_pct:
            continue
        sum_med = sum(med(per_pct[g]) for g in EMD_GROUPS)
        for g in EMD_GROUPS:
            s = qstats(per_pct[g])
            out.append({**c, "group": g, "label": GROUP_LABEL[g], "median_pct": s["median"], "q1_pct": s["q1"],
                        "q3_pct": s["q3"], "min_pct": s["min"], "max_pct": s["max"], "median_ms": med(per_ms[g]),
                        "n_reps": s["n"], "cell_sum_of_medians_pct": sum_med, "is_subrow": False})
        s = qstats(tri_pct)
        out.append({**c, "group": "tridiagonal_solve", "label": "of which: tridiagonal solve (scipy solve_banded)",
                    "median_pct": s["median"], "q1_pct": s["q1"], "q3_pct": s["q3"], "min_pct": s["min"],
                    "max_pct": s["max"], "median_ms": None, "n_reps": s["n"], "cell_sum_of_medians_pct": sum_med,
                    "is_subrow": True})
        # fine-grained table: total time, call count, mean time per call
        fines = sorted({f for r in runs for f in r["stages"]})
        for f in fines:
            ex = [r["stages"][f][0] for r in runs if f in r["stages"]]
            ca = [r["stages"][f][2] for r in runs if f in r["stages"]]
            tot_run = [sum(v[0] for v in r["stages"].values()) for r in runs if f in r["stages"]]
            per_call = [e / c_ / 1e3 for e, c_ in zip(ex, ca) if c_]
            fine_out.append({**c, "fine_stage": f, "group": group_of(f), "median_total_ms": med(e / 1e6 for e in ex),
                             "median_calls": med(ca), "median_us_per_call": med(per_call),
                             "median_pct_of_total": med(100.0 * e / t for e, t in zip(ex, tot_run) if t)})
    return out, fine_out


# ----- 3. closure / instrumentation overhead ------------------------------------------------ #

def agg_closure(d, runtime_rows):
    unprof = {ckey(r): r["median_s"] for r in runtime_rows}
    unprof_min = {ckey(r): r["min_s"] for r in runtime_rows}
    extra_ns = (d.diag.get("timer_overhead") or {}).get("wrapped_call_extra_ns")
    cprof_wall = defaultdict(list)
    for r in d.cprof_runs:
        cprof_wall[ckey(r)].append(r["wall_ns"] / 1e9)
    out = []
    for k, runs in stage_cells(d).items():
        instr = [r["total_ns"] / 1e9 for r in runs]
        sums = [sum(v[0] for v in r["stages"].values()) / r["total_ns"] for r in runs if r["total_ns"]]
        ev = [r["events"] for r in runs]
        mi, mu = med(instr), unprof.get(k)
        mi_min, mu_min = min(instr), unprof_min.get(k)
        row = {**cell_dict(k), "unprofiled_median_ms": mu * 1e3 if mu else None, "instrumented_median_ms": mi * 1e3,
               "instrumentation_overhead_pct": 100.0 * (mi / mu - 1.0) if mu else None,
               "unprofiled_min_ms": mu_min * 1e3 if mu_min else None, "instrumented_min_ms": mi_min * 1e3,
               "instrumentation_overhead_min_pct": 100.0 * (mi_min / mu_min - 1.0) if mu_min else None,
               "stages_sum_over_total_pct": 100.0 * med(sums), "uncategorised_remainder_pct": 100.0 * (1.0 - med(sums)),
               "timer_events_per_run": med(ev),
               "est_timer_cost_pct": (100.0 * med(ev) * extra_ns / 1e9 / mi) if extra_ns and mi else None,
               "replica_used": bool(runs[0].get("replica")), "n_reps": len(runs)}
        cw = cprof_wall.get(k)
        row["cprofile_wall_ms"] = med(cw) * 1e3 if cw else None
        row["cprofile_inflation_x"] = (min(cw) / mu_min) if cw and mu_min else None      # min/min: robust to positive noise
        out.append(row)
    return out


# ----- 4. cProfile ---------------------------------------------------------------------------- #

def agg_cprofile(d, runtime_rows):
    unprof = {ckey(r): r["min_s"] for r in runtime_rows}
    cells = defaultdict(list)
    for r in d.cprof_runs:
        cells[ckey(r)].append(r)
    split = []
    for k in sorted(cells, key=sort_key):
        runs = cells[k]
        pct = defaultdict(list)
        for r in runs:
            tot = r["total_self_s"] or 1e-12
            for cat in CPROF_CATS:
                pct[cat].append(100.0 * r["by_cat_s"].get(cat, 0.0) / tot)
        row = {**cell_dict(k), "n_runs": len(runs)}
        for cat in CPROF_CATS:
            row[cat + "_pct"] = med(pct[cat])
        row["py_frames_total_pct"] = sum(row[c + "_pct"] for c in CPROF_CATS if c.startswith("py_frame"))
        row["interpreter_side_upper_bound_pct"] = 100.0 - row["compiled_visible_pct"]
        mu = unprof.get(k)
        row["cprofile_inflation_x"] = min(r["wall_ns"] / 1e9 for r in runs) / mu if mu else None   # min/min
        split.append(row)
    top = []
    for r in d.cprof_fn:
        rows = sorted(r["rows"], key=lambda x: -x[5])
        tot = sum(x[5] for x in rows) or 1e-12
        for rank, x in enumerate(rows[:15], 1):
            top.append({**{k: r[k] for k in CELL_KEYS}, "rank": rank, "function": x[2], "file": x[0], "line": x[1],
                        "ncalls": x[3], "tottime_s": x[5], "cumtime_s": x[6], "pct_self": 100.0 * x[5] / tot,
                        "category": x[7]})
    return split, top


# ----- 5. fixed per-call overhead model (interpreter/dispatch) vs compiled work ------------------- #

def agg_overhead_model(d, cprof_split):
    loops = {ckey(r): r["py_builtin_iter_pct"] for r in cprof_split}
    upper = {ckey(r): r["interpreter_side_upper_bound_pct"] for r in cprof_split}
    out = []
    import numpy as np
    for k, runs in stage_cells(d, "emd").items():
        floors, drift = None, None
        snaps = [s["data"] for s in d.floor_unit.get(k, [])]
        if snaps:                                            # floors measured right before/after this unit's stage runs
            flat = [x for s in snaps for x in (s["before"], s["after"])]
            floors = {st_: float(np.mean([x[st_] for x in flat if st_ in x])) for st_ in set().union(*flat)}
            ratios = [s["after"][st_] / s["before"][st_] for s in snaps for st_ in s["before"] if st_ in s["after"] and s["before"][st_] > 0]
            drift = 100.0 * float(np.median(np.abs(np.array(ratios) - 1.0))) if ratios else None
        else:                                                # fall back to the once-per-worker calibration
            floors = (d.floor.get(k[2]) or {}).get(str(FLOOR_N))
        if not floors:
            continue
        fixed_pct, per_stage_pct = [], defaultdict(list)
        for r in runs:
            tot = sum(v[0] for v in r["stages"].values())
            if tot <= 0:
                continue
            fixed = 0.0
            for fine, (ex, _inc, calls) in r["stages"].items():
                fl = floors.get(fine)
                if fl is None:
                    continue
                units = r["n_sift"] if fine == "outer_loop" else calls
                part = min(fl * units, ex)
                fixed += part
                per_stage_pct[group_of(fine)].append(100.0 * part / ex if ex else 0.0)
            fixed_pct.append(100.0 * fixed / tot)
        if not fixed_pct:
            continue
        fx = med(fixed_pct)
        lp = loops.get(k, 0.0) or 0.0
        out.append({**cell_dict(k), "floor_source": "per-unit (before/after)" if snaps else "per-worker", "floor_drift_within_unit_pct": drift,
                    "fixed_per_call_overhead_pct": fx, "python_element_loops_pct": lp,
                    "compiled_numerical_pct": max(0.0, 100.0 - fx - lp), "interpreter_side_model_pct": min(100.0, fx + lp),
                    "cprofile_interpreter_side_upper_bound_pct": upper.get(k),
                    **{f"fixed_share_of_{g}_pct": med(per_stage_pct[g]) for g in EMD_GROUPS if per_stage_pct.get(g)}})
    return out


# ----- 6. Amdahl ------------------------------------------------------------------------------- #

def agg_amdahl(d):
    out = []
    for k, runs in stage_cells(d).items():
        if k[0] != "pinned":
            continue
        f_by, F_dp, F_dpm = defaultdict(list), [], []
        for r in runs:
            ex = {fine: v[0] for fine, v in r["stages"].items()}
            tot = sum(v for fine, v in ex.items() if group_of(fine) in EMD_GROUPS)
            if tot <= 0:
                continue
            a = b = 0.0
            for key, _label, fines, cls, _why in AMDAHL_STAGES:
                f = sum(ex.get(x, 0) for x in fines) / tot
                f_by[key].append(f)
                if cls == "DP":
                    a += f
                if cls in ("DP", "MIXED"):
                    b += f
            F_dp.append(a)
            F_dpm.append(b)
        if not F_dp:
            continue
        for key, label, _fines, cls, why in AMDAHL_STAGES:
            fm = med(f_by[key])
            out.append({**cell_dict(k), "stage": key, "label": label, "class": cls, "f_pct": 100 * fm,
                        "max_speedup_if_stage_free": 1.0 / (1.0 - fm) if fm < 1 else float("inf"), "why": why})
        for key, label, vals in (("ALL_DATA_PARALLEL", "all data-parallel (DP) stages together", F_dp),
                                 ("ALL_DP_AND_MIXED", "DP + MIXED stages together (optimistic)", F_dpm)):
            Fm = med(vals)
            out.append({**cell_dict(k), "stage": key, "label": label, "class": "DP" if key == "ALL_DATA_PARALLEL" else "DP+MIXED",
                        "f_pct": 100 * Fm, "max_speedup_if_stage_free": 1.0 / (1.0 - Fm) if Fm < 1 else float("inf"),
                        "speedup_if_stages_10x": 1.0 / ((1 - Fm) + Fm / 10.0), "speedup_if_stages_100x": 1.0 / ((1 - Fm) + Fm / 100.0),
                        "why": "sum of the stage fractions in the set; every other stage unchanged"})
    return out


# ----- 7. ensembles ------------------------------------------------------------------------------ #

def agg_ensembles(d):
    levels, counts = [], []
    for k, runs in stage_cells(d).items():
        if k[1] not in ("eemd", "ceemdan"):
            continue
        acc = defaultdict(list)
        for r in runs:
            st, tot = r["stages"], r["total_ns"]
            gt = group_times(st)
            emd_t = sum(gt.get(g, 0) for g in EMD_GROUPS)
            all_t = sum(v[0] for v in st.values()) or 1
            acc["noise_generation_pct"].append(100.0 * gt.get("noise_generation", 0) / all_t)
            acc["emd_calls_pct"].append(100.0 * emd_t / all_t)
            acc["ensemble_glue_pct"].append(100.0 * gt.get("ensemble_glue", 0) / all_t)
            inc = lambda name: st.get(name, [0, 0, 0])[1]  # noqa: E731
            if k[1] == "eemd":
                acc["parallelisable_pct"].append(100.0 * inc("trial_update") / tot)
            else:
                nd, fi, ec = inc("phase:noise_decomposition"), inc("phase:first_imf"), inc("phase:end_condition")
                main_only = inc("phase:main_loop") - nd - fi - ec - inc("noise_generation")
                acc["parallelisable_pct"].append(100.0 * (nd + fi) / tot)
                acc["phase_noise_decomposition_pct"].append(100.0 * nd / tot)
                acc["phase_first_imf_pct"].append(100.0 * fi / tot)
                acc["phase_main_loop_serial_pct"].append(100.0 * main_only / tot)
                acc["phase_end_condition_pct"].append(100.0 * ec / tot)
                acc["phase_noise_generation_pct"].append(100.0 * inc("noise_generation") / tot)
        levels.append({**cell_dict(k), **{a: med(v) for a, v in acc.items()}, "n_reps": len(runs)})
        r0 = runs[0]
        counts.append({**cell_dict(k), "emd_calls": r0["emd_calls"],
                       "emd_calls_by_phase": json.dumps(r0["emd_calls_by_phase"]), "imfs_extracted_total": r0["n_imf"],
                       "sift_iterations_total": r0["n_sift"],
                       "mean_sift_iterations_per_imf": r0["n_sift"] / r0["n_imf"] if r0["n_imf"] else None,
                       "output_components": r0["out_rows"], "imfs_hit_max_iteration": r0["capped"]})
    return levels, counts


def agg_parallel(d, runtime_rows, levels):
    idx = {(r["thread_mode"], r["algo"], r["signal"], r["n"], r["trials"], r["procs"]): r
           for r in runtime_rows if r["config"] == "default" and r["algo"] in ("eemd", "ceemdan")}
    fpar = {}
    for r in levels:
        fpar.setdefault((r["algo"], r["signal"], r["n"]), r["parallelisable_pct"] / 100.0)
    pool = d.diag.get("pool_startup_s") or {}
    cores = ((next(iter(d.meta.values()), {}) or {}).get("cpu") or {}).get("available_cores") or max(1, ncores_available())
    out = []
    for (tm, algo, signal, n, trials, procs), r in sorted(idx.items(), key=lambda kv: (ALGO_ORDER[kv[0][1]], kv[0][4], kv[0][5])):
        if procs == 0 or tm != "pinned":
            continue
        ser = idx.get((tm, algo, signal, n, trials, 0))
        p1 = idx.get((tm, algo, signal, n, trials, 1))
        f = fpar.get((algo, signal, n))
        row = {"algo": algo, "signal": signal, "n": n, "trials": trials, "procs": procs, "n_reps": r["n_reps"],
               "median_s": r["median_s"], "q1_s": r["q1_s"], "q3_s": r["q3_s"],
               "serial_median_s": ser["median_s"] if ser else None,
               "speedup_vs_serial": ser["median_s"] / r["median_s"] if ser else None,
               "speedup_vs_pool_1proc": p1["median_s"] / r["median_s"] if p1 else None,
               "efficiency_vs_serial": (ser["median_s"] / r["median_s"] / procs) if ser else None,
               "parallelisable_fraction_measured": f,
               "amdahl_predicted_speedup": (1.0 / ((1 - f) + f / procs)) if f is not None else None,
               "amdahl_ceiling_infinite_procs": (1.0 / (1 - f)) if f is not None and f < 1 else None,
               "pool_startup_s": pool.get(str(procs)), "pool_startup_empty_s": pool.get(f"empty_{procs}"), "cores_available": cores}
        out.append(row)
    return out


def agg_threads(runtime_rows):
    idx = {(r["algo"], r["config"], r["signal"], r["n"], r["trials"], r["procs"], r["thread_mode"]): r for r in runtime_rows}
    out = []
    for (algo, config, signal, n, trials, procs, tm), r in idx.items():
        if tm != "default" or procs != 0:
            continue
        p = idx.get((algo, config, signal, n, trials, procs, "pinned"))
        if p:
            out.append({"algo": algo, "config": config, "signal": signal, "n": n, "trials": trials,
                        "pinned_median_s": p["median_s"], "default_median_s": r["median_s"],
                        "ratio_default_over_pinned": r["median_s"] / p["median_s"],
                        "difference_pct": 100.0 * (r["median_s"] / p["median_s"] - 1.0),
                        "pinned_iqr_pct_of_median": 100.0 * p["iqr_s"] / p["median_s"]})
    return sorted(out, key=lambda r: (ALGO_ORDER[r["algo"]], r["config"], SIGNALS.index(r["signal"]), r["trials"], r["n"]))


def agg_iterations(d, runtime_rows):
    rt = {ckey(r): r for r in runtime_rows}
    out = []
    for k, runs in stage_cells(d, "emd").items():
        r0 = runs[0]
        sc = (r0.get("sift_counts") or [[]])[0] if r0.get("sift_counts") else []
        row = {**cell_dict(k), "output_rows": r0["out_rows"], "imfs_extracted": r0["n_imf"],
               "sift_iterations_total": r0["n_sift"], "iterations_per_imf": ";".join(str(x) for x in sc),
               "max_iterations_single_imf": max(sc) if sc else None,
               "mean_iterations_per_imf": r0["n_sift"] / r0["n_imf"] if r0["n_imf"] else None,
               "imfs_hit_max_iteration": r0["capped"], "median_runtime_s": rt[k]["median_s"] if k in rt else None}
        same = all(((x.get("sift_counts") or [[]])[0] if x.get("sift_counts") else []) == sc for x in runs)
        row["counts_identical_across_reps"] = bool(same)
        out.append(row)
    return out


def agg_memory(d):
    return [{**{k: r[k] for k in CELL_KEYS}, "peak_traced_MiB": r["peak_bytes"] / 2 ** 20,
             "current_traced_MiB": r["current_bytes"] / 2 ** 20} for r in sorted(d.memory, key=lambda r: sort_key(ckey(r)))]


def agg_lines(d):
    out = []
    for r in d.lines:
        rows = sorted(r["rows"], key=lambda x: -x[4])
        for rank, x in enumerate(rows[:20], 1):
            out.append({**{k: r[k] for k in CELL_KEYS}, "rank": rank, "function": x[0], "file": x[1], "lineno": x[2],
                        "hits": x[3], "time_ms": x[4] * 1e3, "pct_of_function": x[5], "source": x[6]})
    return out


def agg_floors(d):
    out = []
    for config, fl in d.floor.items():
        for stage in sorted(fl.get(str(FLOOR_N), {})):
            out.append({"config": config, "stage": stage,
                        "unit": "per sift iteration" if stage == "outer_loop" else "per call",
                        f"floor_us_at_N{FLOOR_N}": fl[str(FLOOR_N)][stage] / 1e3,
                        f"floor_us_at_N{2 * FLOOR_N}": fl.get(str(2 * FLOOR_N), {}).get(stage, float("nan")) / 1e3})
    return out


def agg_canary(d):
    """Drift of a fixed workload timed at the start, every 10 units, and at the end of every worker."""
    out = []
    by_tm = defaultdict(list)
    for r in d.canary:
        by_tm[r["thread_mode"]].append(r)
    for tm, rows in by_tm.items():
        base = rows[0]["min_ns"]
        for i, r in enumerate(rows):
            out.append({"thread_mode": tm, "index": i, "when": r["when"], "timestamp_utc": r["timestamp_utc"],
                        "median_ms": r["median_ns"] / 1e6, "min_ms": r["min_ns"] / 1e6,
                        "min_vs_first_pct": 100.0 * (r["min_ns"] / base - 1.0)})
    return out


# --------------------------------------------------------------------------- #
# Figures (vector only, fonts >= 10 pt at the size they are drawn)             #
# --------------------------------------------------------------------------- #
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F"]
STAGE_COLOR = {g: OKABE_ITO[i] for i, g in enumerate(EMD_GROUPS)}
SIGNAL_COLOR = {"multitone": "#0072B2", "multitone_noisy": "#D55E00", "chirp_am": "#009E73"}
SIGNAL_MARKER = {"multitone": "o", "multitone_noisy": "s", "chirp_am": "^"}
SIGNAL_SHORT = {"multitone": "tone", "multitone_noisy": "tone + noise", "chirp_am": "chirp + AM"}
TRIAL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 10, "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4.5, "xtick.major.size": 3,
        "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "savefig.transparent": False,
        "axes.unicode_minus": False, "legend.frameon": False, "legend.handlelength": 1.6,
    })
    return plt


def save_fig(fig, c, name):
    path = os.path.join(c.outdir, "figures", f"{name}.{c.fig_format}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format=c.fig_format)
    return path


def thin_flags(count):
    """Which of ``count`` ticks get a label: all if few, else every second one counted from the largest N."""
    step = 1 if count <= 5 else 2
    return [((count - 1 - j) % step == 0) for j in range(count)]


def label_ticks(ax, lengths):
    import numpy as np
    ax.set_xticks(np.arange(len(lengths)))
    ax.set_xticklabels([fmt_n(n) if keep else "" for n, keep in zip(lengths, thin_flags(len(lengths)))])


def panel_tag(ax, tag, right=False):
    ax.text(0.98 if right else 0.02, 0.97, tag, transform=ax.transAxes, va="top", ha="right" if right else "left", fontweight="bold")


def log_axis_labels(ax, which="y"):
    """Plain decimal labels on a log axis; minor ticks 2 and 5 are labelled only when the axis spans < 2 decades.
    Call after the data are plotted (uses the current axis limits)."""
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    span = float(np.log10(hi / lo)) if lo > 0 else 9.0

    def major(v, _pos):
        return f"{v:g}"

    def minor(v, _pos):
        if span >= 2 or v <= 0:
            return ""
        lead = v / 10 ** np.floor(np.log10(v))
        return f"{v:g}" if (abs(lead - 2) < 1e-6 or abs(lead - 5) < 1e-6) else ""
    axis.set_major_formatter(FuncFormatter(major))
    axis.set_minor_formatter(FuncFormatter(minor))
    ax.tick_params(axis=which, which="both", labelsize=10)


def fig_stage_breakdown(c, plt, rows):
    import numpy as np
    sel = [r for r in rows if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default"
           and not r["is_subrow"]]
    signals = [s for s in c.signals if any(r["signal"] == s for r in sel)]
    if not signals:
        return None
    lengths = sorted({r["n"] for r in sel})
    fig, axes = plt.subplots(1, len(signals), figsize=(c.text_width, 3.2), sharey=True, constrained_layout=True, squeeze=False)
    x = np.arange(len(lengths))
    for ax, sig in zip(axes[0], signals):
        M = np.full((len(EMD_GROUPS), len(lengths)), np.nan)
        for i, g in enumerate(EMD_GROUPS):
            for j, n in enumerate(lengths):
                for r in sel:
                    if r["signal"] == sig and r["n"] == n and r["group"] == g:
                        M[i, j] = r["median_pct"]
        col = np.nansum(M, axis=0)
        col[col == 0] = np.nan
        M = 100.0 * M / col                               # medians re-normalised so every bar sums to 100 %
        bottom = np.zeros(len(lengths))
        for i, g in enumerate(EMD_GROUPS):
            v = np.nan_to_num(M[i])
            ax.bar(x, v, bottom=bottom, color=STAGE_COLOR[g], edgecolor="white", linewidth=0.5, width=0.84,
                   label=GROUP_SHORT[g])
            bottom += v
        ax.set_title(SIGNAL_SHORT[sig])
        label_ticks(ax, lengths)
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("share of EMD runtime (%)")
    axes[0][len(signals) // 2].set_xlabel("signal length $N$ (samples)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, columnspacing=1.2, handletextpad=0.5)
    return save_fig(fig, c, "fig1_stage_breakdown")


def _loglog_slope(ns, ys):
    import numpy as np
    if len(ns) < 3:
        return None
    return float(np.polyfit(np.log10(ns), np.log10(ys), 1)[0])


def fig_runtime_vs_length(c, plt, rt):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    emd = [r for r in rt if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default"]
    t0 = min([x["trials"] for x in rt if x["algo"] in ("eemd", "ceemdan")] or [0])
    ens = [r for r in rt if r["thread_mode"] == "pinned" and r["algo"] in ("eemd", "ceemdan") and r["procs"] == 0
           and r["trials"] == t0 and r["config"] == "default"]
    if not emd:
        return None
    ncol = 2 if ens else 1
    fig, axes = plt.subplots(1, ncol, figsize=(c.text_width if ncol == 2 else c.col_width, 3.3), constrained_layout=True, squeeze=False)
    ax = axes[0][0]
    first = None
    lens = sorted({r["n"] for r in emd})
    for sig in c.signals:
        pts = sorted([r for r in emd if r["signal"] == sig], key=lambda r: r["n"])
        if not pts:
            continue
        ns = np.array([r["n"] for r in pts], float)
        m = np.array([r["median_s"] for r in pts])
        lo, hi = m - np.array([r["q1_s"] for r in pts]), np.array([r["q3_s"] for r in pts]) - m
        s = _loglog_slope(ns, m)
        lab = SIGNAL_SHORT[sig] + (f" ($N^{{{s:.2f}}}$)" if s is not None else "")
        ax.errorbar(ns, m, yerr=[lo, hi], marker=SIGNAL_MARKER[sig], color=SIGNAL_COLOR[sig], capsize=2, label=lab)
        first = first or (ns[0], m[0])
    if first:
        xs = np.array(lens, float)
        ax.plot(xs, first[1] * xs / first[0], color="k", ls=":", lw=1.0, label=r"$\propto N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signal length $N$ (samples)")
    ax.set_ylabel("EMD runtime (s)")
    ax.set_xticks(lens)
    ax.set_xticklabels([fmt_n(n) if keep else "" for n, keep in zip(lens, thin_flags(len(lens)))])
    ax.xaxis.set_minor_locator(NullLocator())
    log_axis_labels(ax, "y")
    panel_tag(ax, "(a)")
    handles, labels = ax.get_legend_handles_labels()
    if ens:
        ax2 = axes[0][1]
        for algo, ls in (("eemd", "-"), ("ceemdan", "--")):
            for sig in c.signals:
                pts = sorted([r for r in ens if r["algo"] == algo and r["signal"] == sig], key=lambda r: r["n"])
                if not pts:
                    continue
                ns = np.array([r["n"] for r in pts], float)
                m = np.array([r["median_s"] for r in pts])
                lo, hi = m - np.array([r["q1_s"] for r in pts]), np.array([r["q3_s"] for r in pts]) - m
                ax2.errorbar(ns, m, yerr=[lo, hi], marker=SIGNAL_MARKER[sig], color=SIGNAL_COLOR[sig], ls=ls, capsize=2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("signal length $N$ (samples)")
        ax2.set_ylabel(f"serial runtime (s), {t0} trials")
        elens = sorted({r["n"] for r in ens})
        ax2.set_xticks(elens)
        ax2.set_xticklabels([fmt_n(n) for n in elens])
        ax2.xaxis.set_minor_locator(NullLocator())
        log_axis_labels(ax2, "y")
        panel_tag(ax2, "(b)")
        handles += [Line2D([], [], color="k", ls="-", label="EEMD"), Line2D([], [], color="k", ls="--", label="CEEMDAN")]
        labels += ["EEMD", "CEEMDAN"]
    fig.legend(handles, labels, loc="outside lower center", ncol=4 if ens else 2, columnspacing=1.2, handletextpad=0.5)
    return save_fig(fig, c, "fig2_runtime_vs_length")


def fig_compiled_vs_interpreter(c, plt, ov, cs):
    import numpy as np
    sel = [r for r in ov if r["thread_mode"] == "pinned" and r["config"] == "default"]
    cps = [r for r in cs if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default"]
    signals = [s for s in c.signals if any(r["signal"] == s for r in sel)]
    if not signals:
        return None
    lengths = sorted({r["n"] for r in sel})
    fig, axes = plt.subplots(2, len(signals), figsize=(c.text_width, 5.0), sharey=True, sharex=True, constrained_layout=True, squeeze=False)
    x = np.arange(len(lengths))
    model = [("compiled_numerical_pct", "compiled numerical work", "#0072B2"),
             ("fixed_per_call_overhead_pct", "fixed per-call overhead", "#E69F00"),
             ("python_element_loops_pct", "Python element loops", "#D55E00")]
    cprof = [(("compiled_visible_pct",), "NumPy/SciPy C entry points", "#0072B2"),
             (("py_builtin_iter_pct", "py_builtin_other_pct"), "Python builtins", "#D55E00"),
             (("py_frames_total_pct",), "Python frames (incl. NumPy operator kernels)", "#999999")]
    for j, sig in enumerate(signals):
        for row, (src, layers) in enumerate(((sel, model), (cps, cprof))):
            ax = axes[row][j]
            bottom = np.zeros(len(lengths))
            for keys, lab, col in layers:
                keys = (keys,) if isinstance(keys, str) else keys
                v = np.array([sum(next((r[k] for r in src if r["signal"] == sig and r["n"] == n), np.nan) for k in keys) for n in lengths])
                ax.bar(x, np.nan_to_num(v), bottom=bottom, color=col, edgecolor="white", linewidth=0.5, width=0.84, label=lab)
                bottom += np.nan_to_num(v)
            ax.set_ylim(0, 100)
            if row == 0:
                ax.set_title(SIGNAL_SHORT[sig])
            label_ticks(ax, lengths)
    axes[0][0].set_ylabel("stage-timer model (%)")
    axes[1][0].set_ylabel("cProfile self time (%)")
    axes[1][len(signals) // 2].set_xlabel("signal length $N$ (samples)")
    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[1][0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=2, columnspacing=1.2, handletextpad=0.5)
    return save_fig(fig, c, "fig3_compiled_vs_interpreter")


def fig_ensemble_scaling(c, plt, rt, par):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    ens = [r for r in rt if r["thread_mode"] == "pinned" and r["algo"] in ("eemd", "ceemdan") and r["config"] == "default"
           and r["signal"] == c.ref_signal and r["n"] == c.ref_length]
    if not ens:
        return None
    trials_all = sorted({r["trials"] for r in ens})
    tcol = {t: TRIAL_COLORS[i % len(TRIAL_COLORS)] for i, t in enumerate(trials_all)}
    style = {"eemd": ("-", "o"), "ceemdan": ("--", "s")}
    fig, axes = plt.subplots(1, 3, figsize=(c.text_width, 3.2), constrained_layout=True)
    ax = axes[0]
    for algo in ("eemd", "ceemdan"):
        pts = sorted([r for r in ens if r["algo"] == algo and r["procs"] == 0], key=lambda r: r["trials"])
        if pts:
            ts = np.array([r["trials"] for r in pts], float)
            m = np.array([r["median_s"] for r in pts])
            ax.errorbar(ts, m, yerr=[m - np.array([r["q1_s"] for r in pts]), np.array([r["q3_s"] for r in pts]) - m],
                        color="k", ls=style[algo][0], marker=style[algo][1], capsize=2)
            if algo == "eemd" and len(ts) > 1:
                ax.plot(ts, m[0] * ts / ts[0], color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(trials_all)
    ax.set_xticklabels([str(t) for t in trials_all])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("trials")
    ax.set_ylabel("serial runtime (s)")
    log_axis_labels(ax, "y")
    panel_tag(ax, "(a)")
    procs_all = sorted({r["procs"] for r in par}) if par else []
    for k, ax in ((1, axes[1]), (2, axes[2])):
        if not par:
            ax.text(0.5, 0.5, "no parallel\ndata", ha="center", va="center", transform=ax.transAxes)
            continue
        key = "speedup_vs_serial" if k == 1 else "efficiency_vs_serial"
        for algo in ("eemd", "ceemdan"):
            for t in trials_all:
                pts = sorted([r for r in par if r["algo"] == algo and r["trials"] == t and r[key] is not None], key=lambda r: r["procs"])
                if pts:
                    ax.plot([r["procs"] for r in pts], [r[key] for r in pts], color=tcol[t], ls=style[algo][0], marker=style[algo][1])
            amd = sorted({(r["procs"], r["amdahl_predicted_speedup"]) for r in par if r["algo"] == algo and r["amdahl_predicted_speedup"]})
            if amd:
                xs = np.array([a[0] for a in amd], float)
                ys = np.array([a[1] for a in amd])
                ax.plot(xs, ys if k == 1 else ys / xs, color="0.6", ls=style[algo][0], lw=1.0)
        if k == 1:
            xs = np.array(procs_all, float)
            ax.plot(xs, xs, color="k", ls=":", lw=1.0)
            ax.set_ylabel("speed-up vs serial")
        else:
            ax.axhline(1.0, color="k", ls=":", lw=1.0)
            ax.set_ylabel("parallel efficiency")
            ax.set_ylim(0, 1.15)
        ax.set_xticks(procs_all)
        ax.set_xlabel("processes")
        panel_tag(ax, "(b)" if k == 1 else "(c)")
    handles = [Line2D([], [], color="k", ls="-", marker="o", label="EEMD"), Line2D([], [], color="k", ls="--", marker="s", label="CEEMDAN")]
    handles += [Line2D([], [], color=tcol[t], label=f"{t} trials") for t in trials_all]
    handles += [Line2D([], [], color="k", ls=":", label="ideal"), Line2D([], [], color="0.6", label="Amdahl (measured serial part)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.0, handletextpad=0.5)
    return save_fig(fig, c, "fig4_ensemble_scaling")


# --------------------------------------------------------------------------- #
# LaTeX / CSV exports                                                          #
# --------------------------------------------------------------------------- #

def export_raw_csv(c, d):
    base = os.path.join(c.outdir, "raw", "csv")
    write_csv(os.path.join(base, "raw_timing.csv"),
              [{**{k: r[k] for k in CELL_KEYS}, "warmup": r["warmup"], "rep": r["rep"], "t_s": r["t_ns"] / 1e9,
                "t_ns": r["t_ns"], "out_rows": r["out_rows"]} for r in sorted(d.timing, key=lambda r: (sort_key(ckey(r)), r["warmup"] is False, r["rep"]))])
    runs, longs = [], []
    for r in d.stage:
        runs.append({**{k: r[k] for k in CELL_KEYS}, "rep": r["rep"], "total_ns": r["total_ns"], "timer_events": r["events"],
                     "sift_iterations": r["n_sift"], "imfs_extracted": r["n_imf"], "imfs_hit_max_iteration": r["capped"],
                     "emd_calls": r["emd_calls"], "output_rows": r["out_rows"], "instrumented_emd_copy_used": r["replica"],
                     "patches_missing": ";".join(r.get("patch_missing") or [])})
        for fine, (ex, inc, calls) in sorted(r["stages"].items()):
            longs.append({**{k: r[k] for k in CELL_KEYS}, "rep": r["rep"], "fine_stage": fine, "group": group_of(fine),
                          "exclusive_ns": ex, "inclusive_ns": inc, "calls": calls})
    write_csv(os.path.join(base, "raw_stage_runs.csv"), runs)
    write_csv(os.path.join(base, "raw_stage_times.csv"), longs)
    write_csv(os.path.join(base, "raw_cprofile_runs.csv"),
              [{**{k: r[k] for k in CELL_KEYS}, "rep": r["rep"], "wall_ns": r["wall_ns"], "total_self_s": r["total_self_s"],
                **{f"{cat}_s": r["by_cat_s"].get(cat, 0.0) for cat in CPROF_CATS}} for r in d.cprof_runs])
    write_csv(os.path.join(base, "raw_cprofile_functions.csv"),
              [{**{k: r[k] for k in CELL_KEYS}, "file": x[0], "line": x[1], "function": x[2], "ncalls": x[3],
                "primitive_calls": x[4], "tottime_s": x[5], "cumtime_s": x[6], "category": x[7]}
               for r in d.cprof_fn for x in r["rows"]])
    write_csv(os.path.join(base, "raw_line_profile.csv"),
              [{**{k: r[k] for k in CELL_KEYS}, "function": x[0], "file": x[1], "lineno": x[2], "hits": x[3],
                "time_s": x[4], "pct_of_function": x[5], "source": x[6]} for r in d.lines for x in r["rows"]])
    write_csv(os.path.join(base, "raw_memory.csv"), agg_memory(d))
    write_csv(os.path.join(base, "raw_settings.csv"),
              [{**{k: r[k] for k in CELL_KEYS}, "settings_json": json.dumps(r["data"], default=_json_default)} for r in d.settings])
    with open(os.path.join(c.outdir, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump({"diagnostics": d.diag, "floors_ns_per_unit": d.floor}, fh, indent=2, default=_json_default)


def _by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def export_tables(c, R):
    t = os.path.join(c.outdir, "tables")
    for name in ("runtime", "closure", "cprof_split", "cprof_top", "overhead", "amdahl", "ens_levels", "ens_counts",
                 "parallel", "threads", "iters", "memory", "lines", "floors", "fine", "canary"):
        fn = {"runtime": "runtime_summary", "closure": "instrumentation_check", "cprof_split": "cprofile_split",
              "cprof_top": "cprofile_top_functions", "overhead": "compiled_vs_interpreter", "amdahl": "amdahl_all",
              "ens_levels": "ensemble_levels", "ens_counts": "ensemble_counts", "parallel": "parallel_scaling",
              "threads": "thread_sensitivity", "iters": "emd_iterations", "memory": "memory_peak",
              "lines": "line_profile_top", "floors": "per_call_overhead_floors", "fine": "stage_calls_and_times",
              "canary": "machine_drift_canary"}[name]
        write_csv(os.path.join(t, fn + ".csv"), R.get(name, []))
    write_csv(os.path.join(t, "stage_fractions.csv"), R.get("stage_groups", []))

    # --- runtime (EMD, default config, pinned) ---------------------------------------------------
    emd_rt = _by(R["runtime"], thread_mode="pinned", algo="emd", config="default")
    if emd_rt:
        sigs = [s for s in c.signals if any(r["signal"] == s for r in emd_rt)]
        rows = []
        for n in sorted({r["n"] for r in emd_rt}):
            row = [fmt_n(n)]
            reps = []
            for s in sigs:
                r = next((x for x in emd_rt if x["signal"] == s and x["n"] == n), None)
                row.append("--" if r is None else f"{r['median_s']:.3g} ({r['q1_s']:.3g}--{r['q3_s']:.3g})")
                if r:
                    reps.append(r["n_reps"])
            row.append(f"{min(reps)}" if reps and min(reps) == max(reps) else (f"{min(reps)}--{max(reps)}" if reps else "--"))
            rows.append(row)
        write_latex(os.path.join(t, "runtime_emd.tex"), ["$N$"] + [SIGNAL_LABEL[s] for s in sigs] + ["reps"], rows,
                    note="EMD, default settings, threads pinned to 1: median (Q1--Q3) wall time in seconds; reps = timed repetitions.")

    # --- stage fractions: one table per (algo, config, signal) ----------------------------------------
    sg = R.get("stage_groups", [])
    for algo in ("emd", "eemd", "ceemdan"):
        for config in sorted({r["config"] for r in sg if r["algo"] == algo}):
            for sig in c.signals:
                sub = [r for r in sg if r["algo"] == algo and r["config"] == config and r["signal"] == sig and r["thread_mode"] == "pinned"]
                if not sub:
                    continue
                trials = sorted({r["trials"] for r in sub})[0]
                sub = [r for r in sub if r["trials"] == trials]
                ns = sorted({r["n"] for r in sub})
                rows = []
                order = []
                for g in EMD_GROUPS:
                    order.append(g)
                    if g == "spline_build":
                        order.append("tridiagonal_solve")
                for g in order:
                    row = [Raw("\\quad of which: tridiagonal solve") if g == "tridiagonal_solve" else GROUP_LABEL[g]]
                    for n in ns:
                        r = next((x for x in sub if x["n"] == n and x["group"] == g), None)
                        row.append(f1(r["median_pct"]) if r else "--")
                    rows.append(row)
                tag = f"{algo}_{config}_{sig}" + (f"_trials{trials}" if algo != "emd" else "")
                write_latex(os.path.join(t, f"stage_fractions_{tag}.tex"), ["Stage (median \\%)"] + [fmt_n(n) for n in ns], rows,
                            align="l" + "r" * len(ns),
                            note=f"{algo.upper()} {config} settings, signal {sig}, columns = signal length N; median share (%) of "
                                 + ("EMD runtime" if algo == "emd" else "per-trial EMD time inside the ensemble")
                                 + "; medians need not sum to exactly 100.")

    # --- instrumentation check ----------------------------------------------------------------------------
    cl = _by(R["closure"], thread_mode="pinned", algo="emd", config="default")
    if cl:
        rows = [[SIGNAL_LABEL[r["signal"]], fmt_n(r["n"]), f1(r["unprofiled_median_ms"], 2), f1(r["instrumented_median_ms"], 2),
                 f1(r["instrumentation_overhead_pct"]), f1(r["stages_sum_over_total_pct"], 2), f1(r["cprofile_inflation_x"], 2)]
                for r in sorted(cl, key=lambda r: (SIGNALS.index(r["signal"]), r["n"]))]
        write_latex(os.path.join(t, "instrumentation_check_emd.tex"),
                    ["signal", "$N$", "unprofiled (ms)", "stage-timed (ms)", "overhead (\\%)", "stages / total (\\%)", "cProfile ($\\times$)"],
                    rows, align="llrrrrr",
                    note="Stage-timer instrumentation check; 'stages / total' is the sum of all stage times divided by the instrumented wall time.")

    # --- compiled vs interpreter -------------------------------------------------------------------------------
    ov = _by(R["overhead"], thread_mode="pinned", config="default")
    if ov:
        rows = [[SIGNAL_LABEL[r["signal"]], fmt_n(r["n"]), f1(r["compiled_numerical_pct"]), f1(r["fixed_per_call_overhead_pct"]),
                 f1(r["python_element_loops_pct"]), f1(r["cprofile_interpreter_side_upper_bound_pct"])]
                for r in sorted(ov, key=lambda r: (SIGNALS.index(r["signal"]), r["n"]))]
        write_latex(os.path.join(t, "compiled_vs_interpreter_emd.tex"),
                    ["signal", "$N$", "compiled (\\%)", "fixed per-call (\\%)", "Python loops (\\%)", "cProfile upper bound (\\%)"],
                    rows, align="llrrrr",
                    note="Model: fixed per-call overhead = calls x per-call floor measured on 64-sample inputs; Python loops = builtins max/min/sum "
                         "self time from cProfile; cProfile upper bound = everything cProfile does not attribute to a NumPy/SciPy C entry point.")

    # --- Amdahl --------------------------------------------------------------------------------------------------------
    am = R.get("amdahl", [])
    for algo in ("emd", "eemd", "ceemdan"):
        for config in sorted({r["config"] for r in am if r["algo"] == algo}):
            for sig in c.signals:
                sub = [r for r in am if r["algo"] == algo and r["config"] == config and r["signal"] == sig]
                if not sub:
                    continue
                trials = sorted({r["trials"] for r in sub})[0]
                sub = [r for r in sub if r["trials"] == trials]
                ns = sorted({r["n"] for r in sub})
                rows = []
                for key, label, _fines, cls, _why in AMDAHL_STAGES:
                    row = [label, cls]
                    for n in ns:
                        r = next((x for x in sub if x["n"] == n and x["stage"] == key), None)
                        row.append("--" if r is None else f"{r['f_pct']:.1f} / {r['max_speedup_if_stage_free']:.2f}")
                    rows.append(row)
                rows.append("midrule")
                for key, label, cls in (("ALL_DATA_PARALLEL", "All DP stages", "DP"), ("ALL_DP_AND_MIXED", "All DP + MIXED stages", "DP+M")):
                    row = [label, cls]
                    for n in ns:
                        r = next((x for x in sub if x["n"] == n and x["stage"] == key), None)
                        row.append("--" if r is None else f"{r['f_pct']:.1f} / {r['max_speedup_if_stage_free']:.2f}")
                    rows.append(row)
                tag = f"{algo}_{config}_{sig}" + (f"_trials{trials}" if algo != "emd" else "")
                write_latex(os.path.join(t, f"amdahl_{tag}.tex"), ["Stage", "class"] + [fmt_n(n) for n in ns], rows,
                            align="ll" + "r" * len(ns),
                            note="Cells: f (%) / maximum overall speed-up 1/(1-f) if only that stage (or set of stages) were infinitely fast. DP = data-parallel as written, "
                                 "MIXED = partly sequential but parallelisable with restructuring, SEQ = inherently sequential (classification from reading the source).")

    # --- cProfile top functions (largest N, reference signal, EMD) ------------------------------------------------------
    top = _by(R["cprof_top"], thread_mode="pinned", algo="emd", config="default")
    if top:
        sig = c.ref_signal if any(r["signal"] == c.ref_signal for r in top) else top[0]["signal"]
        nmax = max(r["n"] for r in top if r["signal"] == sig)
        sub = sorted([r for r in top if r["signal"] == sig and r["n"] == nmax], key=lambda r: r["rank"])[:12]
        write_latex(os.path.join(t, "cprofile_top_emd.tex"), ["function", "file", "calls", "self (\\%)", "class"],
                    [[r["function"][:48], f"{r['file']}:{r['line']}"[-40:], str(r["ncalls"]), f1(r["pct_self"]), r["category"]] for r in sub],
                    align="llrrl", note=f"cProfile, EMD default, signal {sig}, N={nmax}: top functions by self time (cProfile inflates small functions).")

    # --- line profile (largest N) ------------------------------------------------------------------------------------------------
    ln = _by(R["lines"], thread_mode="pinned", config="default")
    if ln:
        sig = c.ref_signal if any(r["signal"] == c.ref_signal for r in ln) else ln[0]["signal"]
        nmax = max(r["n"] for r in ln if r["signal"] == sig)
        sub = sorted([r for r in ln if r["signal"] == sig and r["n"] == nmax], key=lambda r: r["rank"])[:10]
        write_latex(os.path.join(t, "line_profile_emd.tex"), ["function", "line", "time (ms)", "\\% of function", "source"],
                    [[r["function"], str(r["lineno"]), f1(r["time_ms"], 1), f1(r["pct_of_function"]), r["source"][:55]] for r in sub],
                    align="lrrrl", note=f"line_profiler, EMD default, signal {sig}, N={nmax}. Line times include time in called functions.")

    # --- ensembles ----------------------------------------------------------------------------------------------------------------------
    lv = R.get("ens_levels", [])
    if lv:
        rows = []
        for r in sorted(lv, key=lambda r: (ALGO_ORDER[r["algo"]], SIGNALS.index(r["signal"]), r["n"])):
            rows.append([r["algo"].upper(), SIGNAL_LABEL[r["signal"]], fmt_n(r["n"]), str(r["trials"]), f1(r["noise_generation_pct"], 2),
                         f1(r["emd_calls_pct"]), f1(r["ensemble_glue_pct"], 2), f1(r["parallelisable_pct"])])
        write_latex(os.path.join(t, "ensemble_levels.tex"),
                    ["algo", "signal", "$N$", "trials", "noise gen. (\\%)", "EMD calls (\\%)", "glue (\\%)", "inside Pool code (\\%)"],
                    rows, align="lllrrrrr",
                    note="Serial run, stage timers: share of EEMD/CEEMDAN time. 'inside Pool code' = share that runs inside pool.map/imap when parallel=True.")
    pr = R.get("parallel", [])
    if pr:
        rows = [[r["algo"].upper(), str(r["trials"]), str(r["procs"]), f"{r['median_s']:.3g}", f1(r["speedup_vs_serial"], 2),
                 f1((r["efficiency_vs_serial"] or 0) * 100, 0), f1(r["amdahl_predicted_speedup"], 2)] for r in pr]
        write_latex(os.path.join(t, "parallel_scaling.tex"),
                    ["algo", "trials", "procs", "median (s)", "speed-up", "efficiency (\\%)", "Amdahl pred."], rows, align="lrrrrrr",
                    note="Speed-up relative to the serial (parallel=False) run; Amdahl prediction from the measured share of time inside Pool code.")


# --------------------------------------------------------------------------- #
# Stage mapping document                                                       #
# --------------------------------------------------------------------------- #

def write_stage_mapping(c, d):
    meta = next(iter(d.meta.values()), None)
    if not meta:
        return
    m = meta["pyemd"]["methods"]
    ver = meta["pyemd"]["version"]
    df = meta["pyemd_defaults"]
    g = lambda k: m.get(k) or "n/a"  # noqa: E731
    lines = [f"# PyEMD stage mapping (PyEMD {ver}, read from the installed source)", "",
             f"Source files hashed in metadata.json: {', '.join(meta['pyemd']['sha256'])}.", "",
             "| Algorithm stage | Where it lives in PyEMD | Timer name(s) |", "|---|---|---|",
             f"| Extrema detection | `EMD.find_extrema` ({g('EMD.find_extrema')}) -> `_find_extrema_simple` ({g('EMD._find_extrema_simple')}) "
             f"[default] or `_find_extrema_parabol` ({g('EMD._find_extrema_parabol')}); called three times per sift iteration: at the loop head, "
             f"inside `extract_max_min_spline`, and after the subtraction (for the stopping test) | `extrema_detection`, `stop_extrema` |",
             f"| Boundary / mirror handling | `EMD.prepare_points` ({g('EMD.prepare_points')}) -> `prepare_points_simple` ({g('EMD.prepare_points_simple')}); `nbsym` extra extrema mirrored at each edge | `boundary_handling` |",
             f"| Spline construction | `EMD.spline_points` ({g('EMD.spline_points')}) -> `splines.cubic` ({g('splines.cubic')}) -> `scipy.interpolate.CubicSpline(X, Y)`; "
             f"tridiagonal solve = `scipy.linalg.solve_banded` inside `CubicSpline.__init__`; `cubic_spline_3pts` ({g('splines.cubic_spline_3pts')}) when only 3 knots exist | `spline_construction`, `tridiagonal_solve`, `spline_3pt` |",
             "| Spline evaluation | `spl(x)` in `splines.cubic`; evaluation grid `t = T[(T>=x0)&(T<=xN)]` in `spline_points` | `spline_evaluation`, `spline_grid_select` |",
             f"| Envelope mean and subtraction | inline in `EMD.emd` ({g('EMD.emd')}), no method: `mean = 0.5*(max_env+min_env)`, `imf_old = imf.copy()`, `imf = imf - mean` | `envelope_mean_subtract` (instrumented copy of `emd`) |",
             f"| Stopping criterion (IMF check) | `EMD.check_imf` ({g('EMD.check_imf')}) plus the extrema/zero-crossing recount inline in `emd` | `stop_check`, `stop_extrema` |",
             f"| Outer EMD loop, bookkeeping, end condition | `EMD.emd` ({g('EMD.emd')}): IMF loop, `np.sum(IMF[:k])`, `np.vstack`, `EMD.end_condition` ({g('EMD.end_condition')}), final residue | `outer_loop`, `end_condition`, `extract_glue` |",
             f"| EEMD: noise / per-trial EMD / aggregation | `EEMD.generate_noise` ({g('EEMD.generate_noise')}), `EEMD._trial_update` ({g('EEMD._trial_update')}) -> `EEMD.emd` -> `EMD.emd`; "
             f"aggregation inline in `EEMD.eemd` ({g('EEMD.eemd')}) and `ensemble_mean` ({g('EEMD.ensemble_mean')}); `multiprocessing.Pool` created per call when `parallel=True` | `noise_generation`, `trial_update`, `ensemble_glue` |",
             f"| CEEMDAN | `CEEMDAN.ceemdan` ({g('CEEMDAN.ceemdan')}): `generate_noise`, `_decompose_noise` ({g('CEEMDAN._decompose_noise')}, Pool if parallel), `_eemd` ({g('CEEMDAN._eemd')}, first IMF, Pool if parallel), "
             f"main per-IMF trial loop (plain Python loop, serial even if `parallel=True`), `end_condition` ({g('CEEMDAN.end_condition')}, one extra EMD call per IMF) | `phase:*` stages |",
             "", "## Defaults in force (read from freshly constructed objects)", "", "```json", json.dumps(df, indent=2, default=str), "```", "",
             "## Notes", "",
             "* Stage times use exclusive (self) time: a stage's children are subtracted, so stage shares add up to the total.",
             "* `envelope_mean_subtract`, the stopping-test extrema recount and the outer-loop bookkeeping have no PyEMD method of their own; they are timed in an "
             "instrumented copy of `EMD.emd` that is checked to be bit-identical to the library's own output (see `diagnostics.json`).",
             "* For spline kinds other than `cubic`, construction and evaluation are split the same way (`akima`, `pchip`, `cubic_hermite`, `interp1d` kinds)."]
    with open(os.path.join(c.outdir, "stage_mapping.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Findings summary (every number is computed from the data)                    #
# --------------------------------------------------------------------------- #

def _rng(vals, nd=0, unit="%"):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}{unit}" if round(lo, nd) == round(hi, nd) else f"{lo:.{nd}f}--{hi:.{nd}f}{unit}"


def _med_iqr(vals, nd=1, unit="%"):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    s = qstats(vals)
    if not s:
        return "n/a"
    return f"median {s['median']:.{nd}f}{unit} (IQR {s['q1']:.{nd}f} to {s['q3']:.{nd}f})"


def _cores(meta):
    n = (meta.get("cpu") or {}).get("available_cores")
    return "? cores" if n is None else f"{n} core{'s' if n != 1 else ''} available"


def _level(p):
    return "dominant" if p >= 50 else ("significant" if p >= 20 else "minor")


def write_findings(c, d, R):
    meta = next(iter(d.meta.values()), {}) or {}
    cpu = (meta.get("cpu") or {}).get("model", "unknown CPU")
    ver = (meta.get("pyemd") or {}).get("version", "?")
    sg = [r for r in R.get("stage_groups", []) if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default" and not r["is_subrow"]]
    rt = [r for r in R["runtime"] if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default"]
    out = []
    study_mode = None
    try:
        with open(os.path.join(c.outdir, "metadata.json"), "r", encoding="utf-8") as fh:
            study_mode = json.load(fh).get("study", {}).get("mode")
    except Exception:                                          # noqa: BLE001
        study_mode = getattr(c, "mode", None)
    if study_mode == "quick":
        out.append("**QUICK MODE (small smoke-test grid): do not quote these numbers in the paper; run `--mode full`.**")
    if rt:
        reps = sorted({r["n_reps"] for r in rt})
        ns = sorted({r["n"] for r in rt})
        iqr_pct = [100.0 * r["iqr_s"] / r["median_s"] for r in rt if r["median_s"]]
        out.append(f"PyEMD {ver}; {cpu}; {_cores(meta)}; threads pinned to 1; "
                   f"EMD default settings; N = {fmt_n(ns[0])}--{fmt_n(ns[-1])}; medians of {reps[0] if len(reps) == 1 else f'{reps[0]}--{reps[-1]}'} repetitions "
                   f"(run-to-run IQR: median {med(iqr_pct):.1f}% of the median).")
    # 1. bottlenecks
    if sg:
        by_g = defaultdict(list)
        for r in sg:
            by_g[r["group"]].append(r["median_pct"])
        top = sorted(by_g, key=lambda g: -med(by_g[g]))[:3]
        txt = "; ".join(f"{GROUP_SHORT[g]} {med(by_g[g]):.0f}% (range {_rng(by_g[g])})" for g in top)
        tri = [r["median_pct"] for r in R["stage_groups"] if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default" and r["group"] == "tridiagonal_solve"]
        am = [r for r in R.get("amdahl", []) if r["algo"] == "emd" and r["config"] == "default" and r["stage"] == "ALL_DATA_PARALLEL"]
        am2 = [r for r in R.get("amdahl", []) if r["algo"] == "emd" and r["config"] == "default" and r["stage"] == "ALL_DP_AND_MIXED"]
        s1 = f"**Top bottlenecks** (median share over all lengths/signals): {txt}."
        if tri:
            s1 += f" Tridiagonal solve alone: {med(tri):.0f}%."
        if am:
            s1 += f" Amdahl ceiling if all data-parallel stages were free: {_rng([r['max_speedup_if_stage_free'] for r in am], 1, 'x')}"
            if am2:
                s1 += f" ({_rng([r['max_speedup_if_stage_free'] for r in am2], 1, 'x')} if boundary handling and the tridiagonal solve are parallelised too)"
            s1 += "."
        out.append(s1)
    # 2. interpreter overhead
    ov = [r for r in R.get("overhead", []) if r["thread_mode"] == "pinned" and r["config"] == "default"]
    if ov:
        n_lo, n_hi = min(r["n"] for r in ov), max(r["n"] for r in ov)
        lo = [r["interpreter_side_model_pct"] for r in ov if r["n"] == n_lo]
        hi = [r["interpreter_side_model_pct"] for r in ov if r["n"] == n_hi]
        ub = [r["cprofile_interpreter_side_upper_bound_pct"] for r in ov if r["n"] == n_hi and r["cprofile_interpreter_side_upper_bound_pct"] is not None]
        out.append(f"**Interpreter overhead** (fixed per-call cost + Python element loops): {_rng(lo)} of runtime at N={fmt_n(n_lo)} "
                   f"({_level(med(lo))}), {_rng(hi)} at N={fmt_n(n_hi)} ({_level(med(hi))}). cProfile gives only an upper bound "
                   f"({_rng(ub)} at N={fmt_n(n_hi)}) because it also counts NumPy operator kernels as Python time.")
    # 3. length dependence
    if sg:
        ns = sorted({r["n"] for r in sg})
        if len(ns) >= 2:
            ch = []
            for g in EMD_GROUPS:
                a = med([r["median_pct"] for r in sg if r["group"] == g and r["n"] == ns[0]])
                b = med([r["median_pct"] for r in sg if r["group"] == g and r["n"] == ns[-1]])
                ch.append((abs(b - a), g, a, b))
            ch.sort(reverse=True)
            s3 = "**Length dependence**: " + "; ".join(f"{GROUP_SHORT[g]} {a:.0f}% -> {b:.0f}%" for _d, g, a, b in ch[:2]) + f" (N={fmt_n(ns[0])} -> {fmt_n(ns[-1])})."
            slopes = []
            for sig in c.signals:
                pts = sorted([r for r in rt if r["signal"] == sig], key=lambda r: r["n"])
                if len(pts) >= 3:
                    slopes.append((sig, _loglog_slope([r["n"] for r in pts], [r["median_s"] for r in pts])))
            if slopes:
                s3 += " Runtime ~ N^s with " + ", ".join(f"s={s:.2f} ({SIGNAL_SHORT[g]})" for g, s in slopes) + "."
            it = [r for r in R.get("iters", []) if r["thread_mode"] == "pinned" and r["config"] == "default"]
            if it:
                a = [r["sift_iterations_total"] for r in it if r["n"] == ns[0]]
                b = [r["sift_iterations_total"] for r in it if r["n"] == ns[-1]]
                s3 += f" Sift iterations per EMD: {_rng(a, 0, '')} -> {_rng(b, 0, '')}."
            out.append(s3)
    # 4. surprises
    sur = []
    red = d.diag.get("redundant_find_extrema")
    if red:
        cp = [v["calls_per_sift_iteration"] for v in red.values() if isinstance(v, dict) and v.get("calls_per_sift_iteration")]
        rf = [100 * v["redundant_fraction"] for v in red.values() if isinstance(v, dict) and v.get("redundant_fraction") is not None]
        if cp:
            sur.append(f"`find_extrema` runs {_rng(cp, 1, 'x')} per sift iteration; {_rng(rf)} of calls received an input identical to an earlier call (N=2000 diagnostic)")
    cs = [r for r in R.get("cprof_split", []) if r["thread_mode"] == "pinned" and r["algo"] == "emd" and r["config"] == "default"]
    if cs:
        nmax = max(r["n"] for r in cs)
        bi = [r["py_builtin_iter_pct"] for r in cs if r["n"] == nmax]
        if bi and max(bi) >= 2:
            sur.append(f"builtin max/min over arrays take {_rng(bi)} of cProfile self time at N={fmt_n(nmax)}")
    ps = d.diag.get("parallel_semantics") or {}
    if "eemd_distinct_first_imf_parallel" in ps:
        sur.append(f"parallel EEMD used {ps['eemd_distinct_first_imf_parallel']} distinct noise realisation(s) for {ps['trials']} trials (serial: {ps.get('eemd_distinct_first_imf_serial')})")
    if "ceemdan_pools_created" in ps:
        lv = [r for r in R.get("ens_levels", []) if r["algo"] == "ceemdan"]
        sur.append(f"CEEMDAN(parallel=True) creates {ps['ceemdan_pools_created']} Pools per call"
                   + (f"; only {_rng([r['parallelisable_pct'] for r in lv])} of its serial time is inside Pool code" if lv else ""))
    pr = [r for r in R.get("parallel", []) if r["speedup_vs_serial"] is not None]
    if pr:
        pmax = max(r["procs"] for r in pr)
        parts = []
        for algo in ("eemd", "ceemdan"):
            x = [r["speedup_vs_serial"] for r in pr if r["algo"] == algo and r["procs"] == pmax]
            if x:
                parts.append(f"{algo.upper()} {_rng(x, 2, 'x')}")
        if parts:
            ps_ = d.diag.get("pool_startup_s") or {}
            start = next(iter(d.meta.values()), {}).get("multiprocessing_start_method")
            tail = f" (Pool start-up incl. imports {ps_[str(pmax)]:.2f} s, start method {start})" if str(pmax) in ps_ and start else ""
            sur.append(f"speed-up vs serial at {pmax} process{'es' if pmax != 1 else ''}: " + ", ".join(parts) + tail)
    cl = [r for r in R.get("closure", []) if r["algo"] == "emd" and r["config"] == "default" and r["thread_mode"] == "pinned"]
    if cl:
        sur.append(f"stage timers change runtime by {_med_iqr([r['instrumentation_overhead_pct'] for r in cl])}, "
                   f"min-based {med([r['instrumentation_overhead_min_pct'] for r in cl if r['instrumentation_overhead_min_pct'] is not None]):.1f}%; "
                   f"stages sum to {_rng([r['stages_sum_over_total_pct'] for r in cl], 1)} of the instrumented total")
    th = R.get("threads", [])
    if th:
        sur.append(f"default-thread vs pinned runtimes: {_med_iqr([r['difference_pct'] for r in th])}")
    cn = [r for r in R.get("canary", []) if r["thread_mode"] == "pinned"]
    if len(cn) >= 2:
        sur.append(f"machine drift (fixed canary workload, min time vs first): {_rng([r['min_vs_first_pct'] for r in cn], 1)}")
    slow = [r for r in R["runtime"] if r.get("slow_run_rule_applied")]
    if slow:
        sur.append(f"{len(slow)} slow cell(s) used reduced repetitions (see runtime_summary.csv)")
    if sur:
        out.append("**Surprises / checks**: " + "; ".join(sur) + ".")
    out.append("**Not verified**: the exact split of Python-frame time between interpreter and operator-dispatched NumPy kernels (bounded, not measured); "
               "the data-parallel/sequential classification (from code reading, not measured); results hold for this machine and software stack only.")
    # keep under 300 words: drop the least important surprise clauses first
    def words(parts):
        return sum(len(p.split()) for p in parts)
    while words(out) > 295 and sur:
        sur.pop()
        out = [p for p in out if not p.startswith("**Surprises")]
        if sur:
            out.insert(len(out) - 1, "**Surprises / checks**: " + "; ".join(sur) + ".")
    body = "# Findings summary\n\n_Generated from the raw data in this folder by pyemd_profile.py; review before quoting._\n\n" + "\n\n".join(out) + "\n"
    with open(os.path.join(c.outdir, "findings_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(body)
    return body


# --------------------------------------------------------------------------- #
# One file to hand over                                                        #
# --------------------------------------------------------------------------- #

def make_review_bundle(c, max_file_mb=8):
    """Zip the output folder (skipping single files > max_file_mb) next to it, so there is exactly one file to send."""
    import zipfile
    root = os.path.abspath(c.outdir)
    out = root + "_review.zip"
    top = os.path.basename(root)
    skipped = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                if os.path.getsize(p) > max_file_mb * 2 ** 20:
                    skipped.append(os.path.relpath(p, root))
                    continue
                z.write(p, os.path.join(top, os.path.relpath(p, root)))
        if skipped:
            z.writestr(os.path.join(top, "BUNDLE_NOTE.txt"),
                       f"Files larger than {max_file_mb} MB were left out of this archive (they stay in the output folder):\n"
                       + "\n".join(skipped) + "\n")
    return out, skipped


# --------------------------------------------------------------------------- #
# report_main                                                                  #
# --------------------------------------------------------------------------- #

def report_main(c):
    d = load_data(c.outdir)
    if not d.timing:
        raise SystemExit(f"No raw data found under {os.path.join(c.outdir, 'raw')}. Run the measurements first.")
    mpath = os.path.join(c.outdir, "metadata.json")
    if c.report_only and os.path.exists(mpath):          # take the study design from the stored metadata, not from CLI defaults
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                prm = json.load(fh).get("parameters", {})
            for key in ("ref_length", "ref_signal", "signals", "ens_lengths", "trials", "ncores"):
                if prm.get(key) is not None:
                    setattr(c, key, prm[key])
            LOG.info("report-only: study design (reference signal/length, signals) read from metadata.json")
        except Exception as exc:                          # noqa: BLE001
            LOG.warning("could not read parameters from metadata.json (%r); using command-line defaults", exc)
    for e in d.errors:
        LOG.warning("unit %s failed during measurement: %s", e.get("unit"), e.get("error"))
    LOG.info("building tables ...")
    R = {}

    def safe(name, fn):
        try:
            return fn()
        except Exception as exc:                               # a failing table must not kill the rest of the report
            LOG.error("report step %s failed: %r\n%s", name, exc, traceback.format_exc())
            return None

    R["runtime"] = safe("runtime", lambda: agg_runtime(d)) or []
    sgf = safe("stage", lambda: agg_stage_groups(d)) or ([], [])
    R["stage_groups"], R["fine"] = sgf
    R["closure"] = safe("closure", lambda: agg_closure(d, R["runtime"])) or []
    cp = safe("cprofile", lambda: agg_cprofile(d, R["runtime"])) or ([], [])
    R["cprof_split"], R["cprof_top"] = cp
    R["overhead"] = safe("overhead", lambda: agg_overhead_model(d, R["cprof_split"])) or []
    R["amdahl"] = safe("amdahl", lambda: agg_amdahl(d)) or []
    ec = safe("ensembles", lambda: agg_ensembles(d)) or ([], [])
    R["ens_levels"], R["ens_counts"] = ec
    R["parallel"] = safe("parallel", lambda: agg_parallel(d, R["runtime"], R["ens_levels"])) or []
    R["threads"] = safe("threads", lambda: agg_threads(R["runtime"])) or []
    R["iters"] = safe("iterations", lambda: agg_iterations(d, R["runtime"])) or []
    R["memory"] = safe("memory", lambda: agg_memory(d)) or []
    R["lines"] = safe("lines", lambda: agg_lines(d)) or []
    R["floors"] = safe("floors", lambda: agg_floors(d)) or []
    R["canary"] = safe("canary", lambda: agg_canary(d)) or []
    safe("raw csv", lambda: export_raw_csv(c, d))
    safe("tables", lambda: export_tables(c, R))
    safe("stage mapping", lambda: write_stage_mapping(c, d))

    LOG.info("drawing figures ...")
    plt = safe("matplotlib", mpl_setup)
    figs = []
    if plt is not None:
        for name, fn in (("fig1", lambda: fig_stage_breakdown(c, plt, R["stage_groups"])),
                         ("fig2", lambda: fig_runtime_vs_length(c, plt, R["runtime"])),
                         ("fig3", lambda: fig_compiled_vs_interpreter(c, plt, R["overhead"], R["cprof_split"])),
                         ("fig4", lambda: fig_ensemble_scaling(c, plt, R["runtime"], R["parallel"]))):
            p = safe(name, fn)
            plt.close("all")
            if p:
                figs.append(p)
            else:
                LOG.info("%s skipped (no data for it in this run)", name)
    body = safe("findings", lambda: write_findings(c, d, R))
    if not os.path.exists(os.path.join(c.outdir, "metadata.json")):
        safe("metadata", lambda: write_metadata(c, build_plan(c), None, now_iso()))
    LOG.info("report written to %s (%d figures)", os.path.abspath(c.outdir), len(figs))
    if body:
        print("\n" + body)
    if not getattr(c, "no_bundle", False):
        bundle = safe("bundle", lambda: make_review_bundle(c))
        if bundle:
            LOG.info("ONE FILE TO SEND FOR REVIEW: %s%s", bundle[0], f"  (left out {len(bundle[1])} file(s) > 8 MB, see BUNDLE_NOTE.txt)" if bundle[1] else "")
    return 0


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        prog="pyemd_profile.py",
        description="Profiling study of PyEMD (EMD-signal): EMD, EEMD, CEEMDAN.  Writes raw data, "
                    "summary tables (CSV + LaTeX), vector figures and a findings summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python pyemd_profile.py --mode quick --outdir results_quick
              python pyemd_profile.py --mode full --estimate
              python pyemd_profile.py --mode full --outdir results_full --custom-emd '{"spline_kind":"akima","MAX_ITERATION":50}' --custom-max-imf 8
              python pyemd_profile.py --mode full --outdir results_full --resume
              python pyemd_profile.py --report-only --outdir results_full
            """))
    p.add_argument("--mode", choices=("quick", "full"), default="quick",
                   help="quick: minutes, small grid.  full: the complete study (hours; see --estimate).")
    p.add_argument("--outdir", default="pyemd_profile_results", help="output directory")

    g = p.add_argument_group("experimental grid (defaults depend on --mode)")
    g.add_argument("--lengths", help="comma list of signal lengths for EMD (full: 1000..128000)")
    g.add_argument("--max-length", type=int, help="truncate every length list to <= this value (quick runs)")
    g.add_argument("--signals", help=f"comma list from {','.join(SIGNALS)}")
    g.add_argument("--custom-emd", help="JSON object (or @file.json) of extra EMD() keyword arguments to profile as "
                                       "configuration 'custom', e.g. '{\"spline_kind\":\"akima\",\"MAX_ITERATION\":50}'")
    g.add_argument("--custom-max-imf", type=int, default=-1, help="max_imf passed to emd() for the custom configuration")
    g.add_argument("--trials", help="EEMD/CEEMDAN trial counts (full: 10,50,100)")
    g.add_argument("--procs", help="process counts for parallel runs; 'all' = all available cores (full: 1,2,4,all)")
    g.add_argument("--ensemble-lengths", help="lengths for the EEMD/CEEMDAN per-trial cost study")
    g.add_argument("--ref-length", type=int, help="length used for the trials/parallel sweeps")
    g.add_argument("--ref-signal", default="multitone_noisy", choices=SIGNALS, help="signal used for the trials/parallel sweeps")
    g.add_argument("--only", choices=("emd", "ensembles"), help="run only the EMD part or only EEMD/CEEMDAN")
    g.add_argument("--thread-modes", help="comma list of: pinned (BLAS/OpenMP threads = 1), default (library defaults)")
    g.add_argument("--seed", type=int, default=20240229, help="seed for the test signals")
    g.add_argument("--noise-seed", type=int, default=12345, help="seed passed to EEMD/CEEMDAN noise_seed() before every run")

    r = p.add_argument_group("repetitions")
    r.add_argument("--warmup", type=int, help="warm-up runs per configuration (quick 1, full 2)")
    r.add_argument("--reps", type=int, help="timed repetitions for EMD (quick 5, full 10)")
    r.add_argument("--reps-ensemble", type=int, help="timed repetitions for EEMD/CEEMDAN (quick 3, full 5)")
    r.add_argument("--stage-reps", type=int, help="stage-timer repetitions for EMD (default: same as --reps)")
    r.add_argument("--cprofile-reps", type=int, help="cProfile repetitions for EMD (quick 1, full 3)")
    r.add_argument("--slow-run-seconds", type=float, default=20.0,
                   help="if one warm-up run takes longer than this, repetitions drop to --reps-slow (recorded in the data)")
    r.add_argument("--reps-slow", type=int, default=5, help="timed repetitions used for slow cells")

    m = p.add_argument_group("which measurements")
    m.add_argument("--no-stage", action="store_true", help="skip stage timers")
    m.add_argument("--no-cprofile", action="store_true", help="skip cProfile")
    m.add_argument("--no-line-profile", action="store_true", help="skip line_profiler")
    m.add_argument("--line-profile-lengths", help="lengths at which line_profiler is run (default: 1k, 16k, 128k in full)")
    m.add_argument("--no-memory", action="store_true", help="skip tracemalloc peak-memory runs")

    x = p.add_argument_group("running")
    x.add_argument("--estimate", action="store_true", help="measure one EMD run per cell, print the expected runtime, exit")
    x.add_argument("--yes", action="store_true", help="do not run the pre-flight estimate / confirmation prompt")
    x.add_argument("--resume", action="store_true", help="continue an interrupted run in --outdir (finished units are skipped)")
    x.add_argument("--overwrite", action="store_true", help="delete previous raw data in --outdir first")
    x.add_argument("--report-only", action="store_true", help="rebuild tables/figures/summary from existing raw data")
    x.add_argument("--no-report", action="store_true", help="measure only; do not build tables/figures")
    x.add_argument("--no-bundle", action="store_true", help="do not write <outdir>_review.zip (the one file to send for review)")

    f = p.add_argument_group("figures")
    f.add_argument("--fig-format", choices=("pdf", "svg"), default="pdf", help="vector format (never bitmaps)")
    f.add_argument("--col-width", type=float, default=3.4, help="single-column figure width in inches")
    f.add_argument("--text-width", type=float, default=7.0, help="double-column figure width in inches")

    p.add_argument("--_worker", dest="worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_thread-mode", dest="worker_thread_mode", default=None, help=argparse.SUPPRESS)
    return p


def parse_args(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    a.argv_forward = list(sys.argv[1:] if argv is None else argv)
    return resolve_config(a)


def spawn_worker(c, thread_mode):
    """Run the measurements in a fresh interpreter so the thread environment is set before NumPy loads."""
    env = os.environ.copy()
    for k in THREAD_ENV_VARS:
        if thread_mode == "pinned":
            env[k] = "1"
        else:
            env.pop(k, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    cmd = [sys.executable, os.path.abspath(__file__)] + c.argv_forward + ["--_worker", "--_thread-mode", thread_mode]
    LOG.info("starting worker: threads=%s", thread_mode)
    return subprocess.call(cmd, env=env)


def write_metadata(c, plan, t_start, t_end):
    meta = OrderedDict()
    meta["study"] = {"script": os.path.basename(__file__), "version": SCRIPT_VERSION, "mode": c.mode,
                     "started_utc": t_start, "finished_utc": t_end, "command": [sys.executable] + sys.argv,
                     "plan_summary": describe_plan(plan, c)}
    meta["parameters"] = {k: v for k, v in vars(c).items()
                          if k in ("lengths", "signals", "warmup", "reps", "reps_ensemble", "stage_reps", "cprofile_reps",
                                   "trials", "procs_list", "ens_lengths", "ref_length", "ref_signal", "thread_modes",
                                   "seed", "noise_seed", "slow_run_seconds", "reps_slow", "line_lengths", "ncores",
                                   "custom_emd_json", "custom_max_imf")}
    meta["thread_modes"] = {}
    for tm in c.thread_modes:
        recs = [r for r in jsonl_load(os.path.join(c.outdir, "raw", tm, "records.jsonl")) if r.get("rec") == "meta"]
        if recs:
            meta["thread_modes"][tm] = recs[-1]["data"]
    with open(os.path.join(c.outdir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_json_default)
    return meta


def main(argv=None):
    c = parse_args(argv)
    if c.worker:
        return worker_main(c)

    os.makedirs(c.outdir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(c.outdir, "run.log"), encoding="utf-8")])
    for noisy in ("PyEMD", "fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR if noisy == "PyEMD" else logging.WARNING)

    if c.report_only:
        return report_main(c)

    if c.mode == "quick":
        print("QUICK MODE: small grid for a smoke test.  The study for the paper is --mode full (see README).", flush=True)
    missing = [(m, why) for m, why in (("line_profiler", "line-level profiles"), ("psutil", "CPU/RAM metadata"),
                                       ("threadpoolctl", "thread pinning/reporting")) if importlib.util.find_spec(m) is None]
    if missing:
        print("NOTE: optional packages missing: " + ", ".join(f"{m} ({why})" for m, why in missing)
              + f"\n      install with:  {os.path.basename(sys.executable)} -m pip install " + " ".join(m for m, _ in missing), flush=True)

    plan = build_plan(c)
    print(describe_plan(plan, c), flush=True)

    if c.estimate:
        est = estimate_plan(plan, c)
        print_estimate(est)
        with open(os.path.join(c.outdir, "estimate.json"), "w", encoding="utf-8") as fh:
            json.dump(est, fh, indent=2)
        return 0

    raw = os.path.join(c.outdir, "raw")
    if os.path.isdir(raw) and any(True for _ in os.scandir(raw)) and not c.resume:
        if c.overwrite:
            shutil.rmtree(raw)
        else:
            raise SystemExit(f"{raw} already contains data. Use --resume to continue that run, "
                             f"--overwrite to start over, or pick another --outdir.")

    if c.mode == "full" and not c.yes:
        est = estimate_plan(plan, c)
        print_estimate(est)
        if est["seconds_total"] > 3600 and sys.stdin.isatty():
            if input("Expected runtime exceeds 1 hour. Continue? [y/N] ").strip().lower() != "y":
                print("Aborted.  Use --mode quick, --max-length or --only to shrink the study.")
                return 1

    t_start = now_iso()
    for tm in c.thread_modes:
        rc = spawn_worker(c, tm)
        if rc != 0:
            LOG.error("worker (threads=%s) exited with status %s; continuing with what was recorded", tm, rc)
    write_metadata(c, plan, t_start, now_iso())
    if not c.no_report:
        report_main(c)
    LOG.info("done.  Results in %s", os.path.abspath(c.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

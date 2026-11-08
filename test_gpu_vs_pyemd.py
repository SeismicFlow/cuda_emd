#!/usr/bin/env python3
"""
test_gpu_vs_pyemd.py -- does the GPU EMD / EEMD / CEEMDAN library MATCH PyEMD, and how much FASTER is it?

Put this file next to emd.py / eemd.py / ceemdan.py / kernels.py and run it:

    python test_gpu_vs_pyemd.py --preset smoke       # ~2 min    plumbing check: is everything wired up?
    python test_gpu_vs_pyemd.py --preset quick       # ~10 min   one of each branch, N up to 16k (+ one 65k GPU-only)
    python test_gpu_vs_pyemd.py --preset standard    # ~1 h      the paper run: N 1k..1M, 100 trials, every branch
    python test_gpu_vs_pyemd.py --preset full        # 2-4 h     more sizes/trials, 300 fuzz cases
    (rough, first run; kernel compilation is a few seconds per variant and CPU time depends on your core count.
     PyEMD references are cached, so a re-run after changing the GPU library only repeats the GPU phase.)

Needs: numpy, scipy, PyEMD (pip install EMD-signal -- do NOT drop PyEMD's own EMD.py/EEMD.py/CEEMDAN.py next to the GPU
files: on Windows/macOS they collide with emd.py/eemd.py/ceemdan.py) and, for the GPU phase, cupy. Optional: matplotlib.

WHAT IS TESTED, AND WHY THIS MUCH (the scope argument for the paper)
--------------------------------------------------------------------
The GPU library is a port, so the right unit of coverage is the CODE PATH, not the parameter value. A path is either
a compile-time variant of the kernel (spline_kind x extrema_detection x nbsym), a run-time stopping branch (default
Huang test, FIXE, FIXE_H, the iteration cap, the two end-condition tests, max_imf), a memory regime (shared-memory
fast path vs. global-memory spill, IMF-slot growth) or an ensemble code path (trial chunking, per-order averaging,
separate_trends, CEEMDAN's stage loop). Every one of those is run at least once; values inside a path (0.05 vs 0.07)
are not, because they cannot reach new code. Seven suites:

  emd-scale      EMD at N = 1 000 ... 1 048 576 (round numbers AND 2^16: the old IMF-slot cliff), then GPU-only beyond:
                 match, speed, overflow flags, on white noise / three tones / chirp.
  eemd-scale     EEMD at N = 1k ... 131k with PyEMD's default 100 trials (CEEMDAN likewise): match with IDENTICAL
  ceemdan-scale  injected noise, speed, flags; GPU-only rows beyond what PyEMD can do within --cpu-budget.
  trials         speed vs. number of trials at fixed N (the GPU runs the trials as one batch, so this is its home turf).
  *-branch       one-factor-at-a-time sweeps over the branches listed above (all 7 splines x 2 detectors, nbsym, stop
                 rules, thresholds, max_imf, T, ensemble options). Small N, so the whole matrix is cheap.
  emd-edge/fuzz  tiny N, odd N, N around the block size, plateaus, exact zeros, constants, DC offsets, 1e-6 / 1e12
                 amplitudes, plus seeded random differential testing (every failing fuzz case is reproducible).
  stats          EEMD/CEEMDAN with each library's OWN noise generator: statistically indistinguishable? (an exact
                 comparison is impossible there, so a permutation energy test is used) + seed/chunk determinism.
  emd-geometry   the SAME decomposition launched with 1, 2, 4, 8 thread blocks per signal (the multi-SM mode) and with
                 every spline kind in multi-block mode: the result must be PyEMD's, bit for bit, whatever the geometry.
  api            the PyEMD drop-in surface: EMD()(S) / 2-D calls / get_imfs_and_trend, the batch call, IMF-buffer
                 growth, seeded EEMD / CEEMDAN against PyEMD with the SAME seed (numpy noise stream), ext_EMD given as a
                 CPU PyEMD.EMD, all_imfs / ensemble_std, exceptions where PyEMD raises, no overflow warning on
                 MAX_ITERATION, selftest() of every kernel variant, and launch-shape auto-tuning determinism.

WHAT "MATCH" MEANS
------------------
* EMD: same number of rows, and for every row  max|gpu - pyemd| / max(row peak, 1e-9 * signal peak)  <= --tol
  (default 1e-8). Also reported: per-IMF sift-iteration counts (the GPU kernel's debug counters vs. a counter wrapped
  around PyEMD's extract_max_min_spline) -- equal counts mean the two codes took the same decisions.
* EEMD / CEEMDAN: the noise is generated ONCE on the host, per trial, from a seed, and injected into both libraries
  (PyEMD via a thin subclass that replaces only the noise source; the GPU via its `noise=` argument), so both do exactly
  the same arithmetic on exactly the same inputs. Per-order trial counts (ensemble_count) must also be identical.
  PyEMD's parallel CEEMDAN adds its trials with Pool.imap_unordered, i.e. in COMPLETION order, so two parallel PyEMD
  runs of one input differ in the last bits (amplified to ~1e-7 with parabol extrema); the reference subclass therefore
  uses the ordered Pool.imap (same parallelism, trial order = PyEMD's serial result).
* Independent of the reference (this is what runs where PyEMD is too slow): no overflow flag, all finite,
  reconstruction sum(rows) == signal, and every IMF satisfies |#extrema - #zero-crossings| <= 1 (PyEMD's own test).

WHAT "SPEED" MEANS
------------------
Wall clock of the PyEMD-compatible call (emd() / eemd() / ceemdan(), numpy in, numpy out) on both sides, same inputs.
GPU: one untimed warm-up call per case (kernel compile / memory pool), then the median of several synchronised calls.
PyEMD: its default settings (parallel=True: all CPU cores for EEMD/CEEMDAN; EMD has no parallel mode). Optional
--serial-baseline adds a parallel=False run. A PyEMD run whose predicted time exceeds --cpu-budget is skipped and
that row falls back to the reference-free checks (clearly marked "gpu-only"). Inputs PyEMD itself rejects are
reported as SKIP (the GPU is not run on them unless --probe-rejected).

TWO PHASES, ONE CACHE
---------------------
The CPU phase (PyEMD references + timings) runs first, before CUDA is touched (PyEMD forks worker processes), and is
cached in ./ref_cache keyed by everything that influences the result. Re-running after you change the GPU library
only redoes the GPU phase. `--phase cpu` / `--phase gpu` run them separately (e.g. on different machines).

OUTPUT (./results/<timestamp>/): summary.md (paper-ready tables), results.json (everything, incl. every case's full
configuration), results.csv, run.log, failures.txt (reproduction commands), plots/*.png.
Exit code: 0 = no FAIL, 1 = at least one FAIL, 2 = harness/environment problem.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS_REF_VERSION = "1"   # bump if the meaning of a cached reference changes
CEEMDAN_REF_ORDER = "trial-order"   # part of the CEEMDAN cache key (v1 references summed in completion order)

try:
    from PyEMD import CEEMDAN as _PyCEEMDAN, EEMD as _PyEEMD, EMD as _PyEMD
    _PYEMD_ERR = None
except Exception as _e:  # reported properly in main()
    _PyEMD = _PyEEMD = _PyCEEMDAN = object
    _PYEMD_ERR = _e

# GPU side, filled in by load_gpu_libs() (lazily, so --phase cpu / --list work on a machine without a GPU)
cp = None
GPUBatchEMD = GPUBatchEEMD = GPUBatchCEEMDAN = EMDOverflowWarning = None
GpuEMD = GpuEEMD = GpuCEEMDAN = gpu_kernels = None

# ======================================================================================================================
# Scope: the branches that are enumerated
# ======================================================================================================================
SPLINES = ["cubic", "akima", "pchip", "cubic_hermite", "slinear", "quadratic", "linear"]
DETECTIONS = ["simple", "parabol"]
EMD_THRESHOLD_KEYS = ("energy_ratio_thr", "std_thr", "svar_thr", "total_power_thr", "range_thr")

# Everything a preset controls. Tier 1 = one representative of every branch; tier 2 = the extended matrix.
PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": dict(
        emd_sizes=[300, 600], emd_families=["noise", "multitone"], emd_gpu_only=[],
        ens_sizes=[300], ens_gpu_only=[], scale_trials=4, gpu_only_trials=4,
        trials_sweep_N=300, trials_sweep=[],
        branch_N_emd=(300,), branch_N_ens=300, branch_trials=4, max_tier=1,
        fuzz=6, stats_runs=0, stats_N=300, stats_trials={"eemd": 10, "ceemdan": 6},
        geometry_N=[3000], repeats=2, cpu_budget=60.0),
    "quick": dict(
        emd_sizes=[1000, 4000, 16000], emd_families=["noise", "multitone"], emd_gpu_only=[65536],
        ens_sizes=[1000, 4000, 16000], ens_gpu_only=[], scale_trials=20, gpu_only_trials=10,
        trials_sweep_N=1000, trials_sweep=[],
        branch_N_emd=(1000, 4000), branch_N_ens=1000, branch_trials=8, max_tier=1,
        fuzz=25, stats_runs=6, stats_N=1000, stats_trials={"eemd": 50, "ceemdan": 20},
        geometry_N=[5000, 20000], repeats=3, cpu_budget=300.0),
    "standard": dict(
        emd_sizes=[1000, 4000, 16000, 32000, 64000, 65536, 131072, 262144, 524288, 1048576],
        emd_families=["noise", "multitone", "chirp_am"], emd_gpu_only=[2097152],
        ens_sizes=[1000, 4000, 16000, 32000, 64000, 65536, 131072], ens_gpu_only=[262144],
        scale_trials=100, gpu_only_trials=20,                    # 100 = PyEMD's own default number of trials
        trials_sweep_N=1000, trials_sweep=[10, 100, 500],
        branch_N_emd=(1000, 4000), branch_N_ens=1000, branch_trials=10, max_tier=2,
        fuzz=100, stats_runs=6, stats_N=1000, stats_trials={"eemd": 100, "ceemdan": 30},
        geometry_N=[5000, 20000, 70000], repeats=5, cpu_budget=900.0),
    "full": dict(
        emd_sizes=[1000, 2048, 4000, 8000, 16000, 32000, 64000, 65536, 131072, 262144, 524288, 1048576],
        emd_families=["noise", "multitone", "chirp_am"], emd_gpu_only=[2097152, 4194304],
        ens_sizes=[1000, 4000, 16000, 32000, 64000, 65536, 131072, 262144], ens_gpu_only=[524288],
        scale_trials=100, gpu_only_trials=50,
        trials_sweep_N=2000, trials_sweep=[10, 50, 100, 500, 1000],
        branch_N_emd=(1000, 4000, 16000), branch_N_ens=1000, branch_trials=20, max_tier=2,
        fuzz=300, stats_runs=8, stats_N=2000, stats_trials={"eemd": 100, "ceemdan": 50},
        geometry_N=[5000, 20000, 70000, 200000], repeats=7, cpu_budget=3600.0),
}

# ======================================================================================================================
# Small utilities
# ======================================================================================================================
class Logger:
    """print + append to run.log; ANSI colour only on an interactive console."""
    COL = {"PASS": "32", "WARN": "33", "FAIL": "31", "SKIP": "36"}

    def __init__(self):
        self.fh = None
        self.color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def open(self, path: Path):
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str = "", end: str = "\n"):
        if self.fh:
            self.fh.write(msg + end)
            self.fh.flush()
        if self.color:
            for k, c in self.COL.items():
                msg = msg.replace(f" {k} ", f" \033[{c}m{k}\033[0m ")
        try:
            print(msg, end=end, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode(), end=end, flush=True)


log = Logger()


class FatalGPUError(RuntimeError):
    """The CUDA context is unusable (sticky error); further GPU cases would only produce misleading failures."""


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def jdump(o) -> str:
    return json.dumps(o, sort_keys=True, default=_json_default)


def clean(o):
    """JSON-safe copy: NaN/inf -> strings, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else ("nan" if math.isnan(o) else ("inf" if o > 0 else "-inf"))
    return o


def fmt_s(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x < 1e-3:
        return f"{x * 1e6:.0f} us"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    if x < 120:
        return f"{x:.2f} s"
    if x < 7200:
        return f"{x / 60:.1f} min"
    return f"{x / 3600:.2f} h"


def fmt_e(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if not math.isfinite(x):
        return str(x)
    return f"{x:.1e}"


def fmt_x(x: Optional[float]) -> str:
    return "-" if x is None else (f"{x:.1f}x" if x < 100 else f"{x:.0f}x")


def median(xs):
    return statistics.median(xs) if xs else None


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------------------------------------------------
def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cpu_usable() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def pyemd_version() -> str:
    try:
        import importlib.metadata as md
        for name in ("EMD-signal", "EMD_signal", "emd-signal"):
            try:
                return md.version(name)
            except md.PackageNotFoundError:
                pass
    except Exception:
        pass
    try:
        import PyEMD
        return str(getattr(PyEMD, "__version__", "unknown"))
    except Exception:
        return "unknown"


def host_info() -> Dict[str, Any]:
    import scipy
    return dict(cpu=cpu_model(), cpu_threads_usable=cpu_usable(), cpu_threads_total=os.cpu_count(),
                python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, pyemd=pyemd_version(),
                mp_start_method=__import__("multiprocessing").get_start_method(allow_none=True) or "default")


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        info["name"] = name.decode() if isinstance(name, bytes) else str(name)
        info["sm_count"] = props.get("multiProcessorCount")
        info["compute_capability"] = f"{props.get('major')}.{props.get('minor')}"
    except Exception as e:
        info["name"] = f"unknown ({e})"
    try:
        free, total = cp.cuda.Device().mem_info
        info["mem_total_gb"] = round(total / 2**30, 1)
        info["mem_free_gb"] = round(free / 2**30, 1)
        info["cupy"] = cp.__version__
        info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        info["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
    except Exception:
        pass
    return info


# ======================================================================================================================
# Signals (all float64, all deterministic functions of (kind, N, seed))
# ======================================================================================================================
SEED = 0   # global offset, --seed


def _rng(*parts) -> np.random.Generator:
    ss = [(zlib.crc32(p.encode()) if isinstance(p, str) else int(p)) & 0xFFFFFFFF for p in parts]
    return np.random.default_rng(np.random.SeedSequence(ss))


def make_signal(kind: str, N: int, seed: int = 0) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    if kind == "noise":                       # white Gaussian: ~N/3 maxima in IMF 1, ~log2(N) IMFs -- the stress case
        return _rng("noise", N, seed).standard_normal(N)
    if kind == "multitone":                   # three tones + linear trend: few IMFs, few extrema (fast path at any N)
        return (np.sin(2 * np.pi * 0.1 * n) + 0.7 * np.sin(2 * np.pi * 0.0173 * n + 0.3)
                + 0.5 * np.sin(2 * np.pi * 0.0031 * n + 1.1) + 0.5 * n / max(N - 1, 1))
    if kind == "chirp_am":                    # non-stationary: linear chirp 0.02 -> 0.17 cycles/sample, AM, slow tone
        ph = 2 * np.pi * (0.02 * n + 0.15 * n * n / (2.0 * max(N, 1)))
        return np.sin(ph) * (1 + 0.5 * np.sin(2 * np.pi * 0.0021 * n)) + 0.3 * np.sin(2 * np.pi * 0.0007 * n)
    if kind == "noisy_mix":                   # a realistic EEMD/CEEMDAN input: tones + trend + broadband noise
        return make_signal("multitone", N) + 0.5 * make_signal("noise", N, seed)
    if kind == "pyemd_demo":                  # the example in PyEMD's own EMD.py (needs T = linspace(0, 2 pi, N))
        T = np.linspace(0, 2 * np.pi, N)
        return np.sin(20 * T * (1 + 0.2 * T)) + T ** 2 + np.sin(13 * T)
    if kind.startswith("edge:"):
        return make_edge_signal(kind[5:], N, seed)
    if kind.startswith("fuzz:"):
        return fuzz_signal(int(kind[5:]))
    raise ValueError(f"unknown signal kind {kind!r}")


def make_T(kind: str, N: int, seed: int = 0) -> Optional[np.ndarray]:
    if kind in (None, "none"):
        return None
    if kind == "2pi":
        return np.linspace(0, 2 * np.pi, N)
    if kind == "lin01":
        return np.linspace(0.0, 1.0, N)
    if kind == "jitter":                      # strictly increasing, NON-uniform
        return np.cumsum(1.0 + 0.3 * _rng("T", N, seed).random(N))
    raise ValueError(f"unknown T kind {kind!r}")


EDGE_SIGNALS = ["constant", "zeros", "ramp", "parabola", "one_cycle", "pure_tone", "square", "quantized", "alternating",
                "impulse", "sparse_spikes", "sawtooth", "random_walk", "dc_offset", "tiny_amp", "huge_amp",
                "close_tones", "exact_zeros", "noise"]


def make_edge_signal(name: str, N: int, seed: int = 0) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    r = _rng("edge", name, N, seed)
    if name == "constant":
        return np.full(N, 3.0)
    if name == "zeros":
        return np.zeros(N)
    if name == "ramp":
        return np.linspace(-1.0, 1.0, N)
    if name == "parabola":
        return np.linspace(-1.0, 1.0, N) ** 2
    if name == "one_cycle":
        return np.sin(2 * np.pi * n / max(N, 1))
    if name == "pure_tone":                   # decomposes exactly; the residue is rounding noise
        return np.sin(2 * np.pi * 0.05 * n)
    if name == "square":                      # flat plateaus everywhere (tie handling, plateau midpoints)
        return np.where(np.sin(2 * np.pi * n / 17.0) >= 0, 1.0, -1.0)
    if name == "quantized":                   # many exactly equal neighbours
        return np.round(3 * r.standard_normal(N)) / 3.0
    if name == "alternating":                 # EVERY interior sample is an extremum: the densest possible case
        return np.where(n % 2 == 0, 1.0, -1.0)
    if name == "impulse":
        x = np.zeros(N)
        x[N // 2] = 1.0
        return x
    if name == "sparse_spikes":               # exact zeros between spikes
        x = np.zeros(N)
        idx = r.choice(N, size=max(1, N // 20), replace=False)
        x[idx] = r.standard_normal(len(idx))
        return x
    if name == "sawtooth":
        return (n % 17) / 17.0
    if name == "random_walk":
        return np.cumsum(r.standard_normal(N))
    if name == "dc_offset":                   # unit noise riding on 1e6
        return 1e6 + r.standard_normal(N)
    if name == "tiny_amp":                    # below PyEMD's absolute range_thr: decomposition ends immediately
        return 1e-6 * r.standard_normal(N)
    if name == "huge_amp":
        return 1e12 * r.standard_normal(N)
    if name == "close_tones":
        return np.sin(2 * np.pi * 0.10 * n) + np.sin(2 * np.pi * 0.105 * n)
    if name == "exact_zeros":                 # 0, 1, 0, -1, ... : exact zero samples between the extrema
        return np.tile(np.array([0.0, 1.0, 0.0, -1.0]), N // 4 + 1)[:N]
    if name == "noise":
        return r.standard_normal(N)
    raise ValueError(f"unknown edge signal {name!r}")


# ----------------------------------------------------------------------------------------------------------------------
# seeded random differential testing ("fuzz"): the signal AND the configuration are functions of one integer
# ----------------------------------------------------------------------------------------------------------------------
def fuzz_N(seed: int) -> int:
    r = _rng("fuzzN", seed + SEED)
    return int(round(math.exp(r.uniform(math.log(24), math.log(2500)))))


def fuzz_signal(seed: int) -> np.ndarray:
    r = _rng("fuzzsig", seed + SEED)
    N = fuzz_N(seed)
    n = np.arange(N, dtype=np.float64)
    kind = r.choice(["tones", "chirp", "noise", "mix", "quantized", "spikes", "walk", "square"])
    x = np.zeros(N)
    if kind in ("tones", "mix"):
        for _ in range(int(r.integers(1, 5))):
            x += r.uniform(0.2, 2.0) * np.sin(2 * np.pi * r.uniform(0.002, 0.45) * n + r.uniform(0, 6.28))
    if kind == "chirp":
        x = np.sin(2 * np.pi * (r.uniform(0.01, 0.1) * n + r.uniform(0.01, 0.2) * n * n / (2.0 * N)))
    if kind in ("noise", "mix"):
        x += r.uniform(0.1, 2.0) * r.standard_normal(N)
    if kind == "quantized":
        x = np.round(r.uniform(1, 8) * (np.sin(2 * np.pi * r.uniform(0.01, 0.2) * n) + 0.3 * r.standard_normal(N)))
    if kind == "spikes":
        idx = r.choice(N, size=max(1, N // int(r.integers(3, 30))), replace=False)
        x[idx] = r.standard_normal(len(idx))
    if kind == "walk":
        x = np.cumsum(r.standard_normal(N))
    if kind == "square":
        x = np.sign(np.sin(2 * np.pi * r.uniform(0.005, 0.2) * n + r.uniform(0, 6.28)) + 1e-9)
    if r.random() < 0.3:
        x += r.uniform(-2, 2) * n / max(N - 1, 1)             # trend
    if r.random() < 0.15:
        x += r.choice([1.0, 100.0, 1e4])                       # DC offset
    return x.astype(np.float64)


def fuzz_config(seed: int) -> Dict[str, Any]:
    r = _rng("fuzzcfg", seed + SEED)
    emd: Dict[str, Any] = dict(spline_kind=str(r.choice(SPLINES)),
                               extrema_detection=str(r.choice(["simple", "simple", "parabol"])))
    nb = int(r.choice([1, 2, 2, 2, 3]))
    if nb != 2:
        emd["nbsym"] = nb
    stop = r.choice(["default"] * 6 + ["FIXE", "FIXE_H"])
    if stop == "FIXE":
        emd["FIXE"] = int(r.choice([3, 8]))
    if stop == "FIXE_H":
        emd["FIXE_H"] = int(r.choice([2, 4]))
    if r.random() < 0.3:
        emd["std_thr"] = float(r.choice([0.05, 0.2, 0.5]))
        emd["svar_thr"] = float(r.choice([1e-4, 1e-3, 1e-2]))
        emd["energy_ratio_thr"] = float(r.choice([0.05, 0.2, 0.5]))
    T = "none"
    if emd["extrema_detection"] == "parabol":
        T = str(r.choice(["none", "lin01", "jitter"]))
    return dict(emd=emd, T=T, max_imf=int(r.choice([-1, -1, -1, -1, 1, 2, 4])))

# ======================================================================================================================
# The test case model
# ======================================================================================================================
@dataclasses.dataclass
class Case:
    id: str
    suite: str                       # emd-scale | eemd-scale | ceemdan-scale | trials | emd-branch | eemd-branch | ...
    algo: str                        # emd | eemd | ceemdan
    N: int
    signal: str                      # key for make_signal()
    sig_seed: int = 0
    emd: Dict[str, Any] = dataclasses.field(default_factory=dict)    # inner-EMD kwargs, identical for both libraries
    ens: Dict[str, Any] = dataclasses.field(default_factory=dict)    # ensemble kwargs (trials, noise_width/epsilon, ...)
    gpu_extra: Dict[str, Any] = dataclasses.field(default_factory=dict)  # GPU-only kwargs (chunk_trials)
    max_imf: int = -1                # the max_imf argument of emd() / eemd() / ceemdan()
    T: str = "none"                  # key for make_T()
    inj_seed: int = 4321             # seed of the injected per-trial noise
    inj_kind: str = "normal"
    ref_policy: str = "always"       # always | budget (skip if predicted too slow) | never (GPU-only size)
    expect_overflow: bool = False    # the case is DESIGNED to hit MAX_ITERATION (a flag is then not a finding)
    checks: Tuple[str, ...] = ()     # reference-independent checks: recon, imf_property
    timing: bool = False             # measure speed (repeated GPU calls, repeated cheap CPU calls)
    tier: int = 1
    ext_emd: bool = False            # pass the inner EMD as ext_EMD instead of through kwargs
    note: str = ""

    @property
    def trials(self) -> Optional[int]:
        return self.ens.get("trials")


def _default_stop_rule(emd: Dict[str, Any]) -> bool:
    return not (emd.get("FIXE") or emd.get("FIXE_H") or "MAX_ITERATION" in emd) and \
        emd.get("extrema_detection", "simple") == "simple"


def _emd_checks(emd: Dict[str, Any]) -> Tuple[str, ...]:
    return ("recon", "imf_property") if _default_stop_rule(emd) else ("recon",)


# ----------------------------------------------------------------------------------------------------------------------
# plan builders
# ----------------------------------------------------------------------------------------------------------------------
def plan_emd_scale(P) -> List[Case]:
    out = []
    for fam in P["emd_families"]:
        for N in P["emd_sizes"]:
            out.append(Case(f"emd-scale/{fam}@{N}", "emd-scale", "emd", N, fam, SEED, ref_policy="budget",
                            timing=True, checks=("recon", "imf_property"), inj_seed=4321 + SEED))
    for N in P["emd_gpu_only"]:
        out.append(Case(f"emd-scale/noise@{N}", "emd-scale", "emd", N, "noise", SEED, ref_policy="never",
                        timing=True, checks=("recon", "imf_property"), inj_seed=4321 + SEED))
    return out


def _ens_params(algo: str, trials: int, **over) -> Dict[str, Any]:
    base = dict(trials=trials, noise_width=0.05) if algo == "eemd" else dict(trials=trials, epsilon=0.005)
    base.update(over)
    return base


def plan_ens_scale(P, algo: str) -> List[Case]:
    out = []
    chk = ("recon",) if algo == "ceemdan" else ()
    for N in P["ens_sizes"]:
        out.append(Case(f"{algo}-scale/noisy_mix@{N}", f"{algo}-scale", algo, N, "noisy_mix", SEED,
                        ens=_ens_params(algo, P["scale_trials"]), ref_policy="budget", timing=True, checks=chk,
                        inj_seed=4321 + SEED))
    for N in P["ens_gpu_only"]:      # fewer trials: CEEMDAN keeps a (orders x trials x N) float64 block in host RAM
        out.append(Case(f"{algo}-scale/noisy_mix@{N}", f"{algo}-scale", algo, N, "noisy_mix", SEED,
                        ens=_ens_params(algo, P["gpu_only_trials"]), ref_policy="never", timing=True, checks=chk,
                        inj_seed=4321 + SEED))
    return out


def plan_trials(P) -> List[Case]:
    out = []
    N = P["trials_sweep_N"]
    for algo in ("eemd", "ceemdan"):
        for tr in P["trials_sweep"]:
            out.append(Case(f"trials/{algo}/noisy_mix@{N}/trials={tr}", "trials", algo, N, "noisy_mix", SEED,
                            ens=_ens_params(algo, tr), ref_policy="always", timing=True,
                            checks=("recon",) if algo == "ceemdan" else (), inj_seed=4321 + SEED))
    return out


def plan_emd_branch(P) -> List[Case]:
    Ns = P["branch_N_emd"]
    N0 = Ns[0]
    out: List[Case] = []

    def add(label, emd=None, sig="noise", N=None, tier=1, T="none", max_imf=-1, expect=False, note=""):
        emd = dict(emd or {})
        N = N or N0
        out.append(Case(f"emd-branch/{label}/{sig}@{N}" + (f"/T={T}" if T != "none" else "") +
                        (f"/max_imf={max_imf}" if max_imf != -1 else ""),
                        "emd-branch", "emd", N, sig, SEED, emd=emd, max_imf=max_imf, T=T, expect_overflow=expect,
                        checks=_emd_checks(emd) if max_imf == -1 else ("recon",), tier=tier, note=note))

    # 1) every compile-time variant of the kernel: 7 splines x 2 extrema detectors
    for sp in SPLINES:
        for det in DETECTIONS:
            e = dict(spline_kind=sp, extrema_detection=det)
            add(f"{sp}+{det}", e, "noise", N0, tier=1)
            add(f"{sp}+{det}", e, "chirp_am", N0, tier=2)
            for N in Ns[1:]:    # larger N: the global-memory SPILL path (more extrema than the shared-memory arrays)
                add(f"{sp}+{det}", e, "noise", N, tier=1 if sp == "cubic" else 2)
    # 2) boundary mirroring depth (a compile-time constant)
    for nb in (1, 3, 4):
        for det in DETECTIONS:
            add(f"nbsym={nb}+{det}", dict(nbsym=nb, extrema_detection=det), "noise", N0,
                tier=1 if (nb == 3 and det == "simple") else 2)
    # 3) stopping rules (run-time branches inside the sift loop)
    add("FIXE=5", dict(FIXE=5), tier=1)
    add("FIXE=12", dict(FIXE=12), tier=2)
    add("FIXE_H=2", dict(FIXE_H=2), tier=1)
    add("FIXE_H=5", dict(FIXE_H=5), tier=2)
    add("conv=svar-only", dict(std_thr=0.0, energy_ratio_thr=0.0), tier=1, note="only the scaled-variance test can fire")
    add("conv=std-only", dict(svar_thr=0.0, energy_ratio_thr=0.0), tier=1, note="only the std test can fire")
    add("conv=energy-only", dict(svar_thr=0.0, std_thr=0.0), tier=1, note="only the energy-ratio test can fire")
    add("conv=loose", dict(svar_thr=0.1, std_thr=1.0, energy_ratio_thr=1.0), tier=2)
    add("conv=tight", dict(svar_thr=1e-6, std_thr=1e-3, energy_ratio_thr=1e-3), tier=2, note="many sifts per IMF")
    add("end=range-only", dict(range_thr=3.0, total_power_thr=0.0), tier=1, note="only the range test can end the EMD")
    add("end=power-only", dict(range_thr=0.0, total_power_thr=300.0), tier=1, note="only the L1-power test can end the EMD")
    add("end=never", dict(range_thr=1e-9, total_power_thr=1e-9), tier=2, note="decompose until the residue is a trend")
    add("cap=5", dict(MAX_ITERATION=5), tier=1, expect=True, note="MAX_ITERATION reached on purpose (flag expected)")
    add("cap=50", dict(MAX_ITERATION=50), tier=2)
    add("never-converge", dict(svar_thr=0.0, std_thr=0.0, energy_ratio_thr=0.0, MAX_ITERATION=30), tier=2, expect=True,
        note="no convergence test can fire: every IMF runs to the cap")
    add("max_imf=1", max_imf=1, tier=2)
    add("max_imf=3", max_imf=3, tier=1)
    for sig, tier in (("chirp_am", 2),):     # a second signal for the stop-rule branches
        add("FIXE_H=2", dict(FIXE_H=2), sig, tier=tier)
        add("conv=std-only", dict(svar_thr=0.0, energy_ratio_thr=0.0), sig, tier=tier)
        add("end=range-only", dict(range_thr=3.0, total_power_thr=0.0), sig, tier=tier)
    # 4) the time axis: ignored by "simple", used (normalised) by "parabol"
    add("T+parabol", dict(extrema_detection="parabol"), "pyemd_demo", T="2pi", tier=1)
    add("T+simple", dict(extrema_detection="simple"), "pyemd_demo", T="2pi", tier=2, note="simple must IGNORE T")
    add("T+parabol", dict(extrema_detection="parabol"), "chirp_am", T="jitter", tier=2, note="non-uniform time axis")
    add("T+parabol+akima", dict(extrema_detection="parabol", spline_kind="akima"), "noise", T="lin01", tier=2)
    # 5) PyEMD-compat details
    add("spline=Akima(case)", dict(spline_kind="Akima"), tier=2, note="spline_kind is case-insensitive in PyEMD")
    add("scale_factor=2", dict(scale_factor=2.0), tier=2, note="accepted, no effect (as in PyEMD)")
    return out


def plan_ens_branch(P, algo: str) -> List[Case]:
    N0, tr = P["branch_N_ens"], P["branch_trials"]
    out: List[Case] = []
    base_chk = ("recon",) if algo == "ceemdan" else ()

    def add(label, ens=None, emd=None, sig="noisy_mix", N=None, tier=1, T="none", max_imf=-1, gpu_extra=None,
            ext_emd=False, trials=None, note=""):
        e = _ens_params(algo, trials or tr, **(ens or {}))
        out.append(Case(f"{algo}-branch/{label}/{sig}@{N or N0}", f"{algo}-branch", algo, N or N0, sig, SEED, emd=dict(emd or {}),
                        ens=e, gpu_extra=dict(gpu_extra or {}), max_imf=max_imf, T=T, ext_emd=ext_emd,
                        checks=base_chk if max_imf == -1 else (), tier=tier, inj_seed=4321 + SEED, note=note))

    # the inner EMD in every variant (the ensemble code treats it as a black box, but CEEMDAN drives it with max_imf=1)
    core = {("cubic", "simple"), ("cubic", "parabol"), ("akima", "simple"), ("linear", "simple")}
    for sp in SPLINES:
        for det in DETECTIONS:
            add(f"{sp}+{det}", emd=dict(spline_kind=sp, extrema_detection=det), tier=1 if (sp, det) in core else 2)
    add("nbsym=3", emd=dict(nbsym=3), tier=2)
    add("FIXE_H=3", emd=dict(FIXE_H=3), tier=2)
    add("ext_EMD(akima)", emd=dict(spline_kind="akima"), ext_emd=True, tier=2, note="inner EMD passed as ext_EMD")
    add("max_imf=3", max_imf=3, tier=1)
    add("trials=1", trials=1, tier=1, note="degenerate ensemble")
    add("trials=7,chunk=3", trials=7, gpu_extra=dict(chunk_trials=3), tier=1, note="chunks of 3+3+1 (GPU-side batching)")
    add("trials=10,chunk=1", gpu_extra=dict(chunk_trials=1), tier=2)
    add("T=2pi+parabol", emd=dict(extrema_detection="parabol"), sig="pyemd_demo", T="2pi", tier=2)
    for sig, tier in (("noise", 1), ("multitone", 2), ("chirp_am", 2)):
        add("signal", sig=sig, tier=tier,
            note="white noise: trials differ in IMF count, so per-order averaging over a varying number of trials is exercised"
            if sig == "noise" else "")
    if algo == "eemd":
        add("separate_trends", ens=dict(separate_trends=True), tier=1)
        add("separate_trends", ens=dict(separate_trends=True), sig="noise", tier=2)
        add("noise_width=0", ens=dict(noise_width=0.0), tier=1, note="no noise: must equal plain EMD")
        add("noise_width=0.3", ens=dict(noise_width=0.3), tier=2)
        add("trials=2", trials=2, tier=2)
    else:
        add("beta_progress=False", ens=dict(beta_progress=False), tier=1)
        add("beta_progress=False,noise_scale=0.5", ens=dict(beta_progress=False, noise_scale=0.5), tier=2)
        add("epsilon=0.05", ens=dict(epsilon=0.05), tier=1)
        add("epsilon=0.2", ens=dict(epsilon=0.2), tier=2)
        add("ctor_max_imf=3", ens=dict(max_imf=3), tier=1, note="bound on the stage loop")
        add("call_max_imf=1", max_imf=1, tier=2)
        add("loose_end_thr", ens=dict(range_thr=0.5, total_power_thr=50.0), tier=1,
            note="range_thr / total_power_thr feed CEEMDAN's own end condition AND the inner EMD (as in PyEMD)")
        add("trials=4", trials=4, tier=2)
    return out


def plan_emd_edge(P) -> List[Case]:
    out: List[Case] = []

    def add(sig, N, tier, emd=None, label=None):
        emd = dict(emd or {})
        out.append(Case(f"emd-edge/{label or 'default'}/{sig}@{N}", "emd-edge", "emd", N, f"edge:{sig}", SEED,
                        emd=emd, checks=_emd_checks(emd), tier=tier))

    for N in (2, 3, 4, 5, 7, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025):
        add("noise", N, 1 if N in (2, 3, 5, 8, 33, 127, 128, 129, 1023, 1024, 1025) else 2)
    for sig in EDGE_SIGNALS:
        if sig == "noise":
            continue
        add(sig, 257, 1)                       # odd prime: no accidental alignment with block / warp size
        add(sig, 64, 2)
        add(sig, 1000, 2)
    for sig in ("square", "quantized", "alternating", "sparse_spikes", "exact_zeros"):
        add(sig, 257, 2, dict(spline_kind="akima", extrema_detection="parabol"), "akima+parabol")
        add(sig, 257, 2, dict(spline_kind="pchip"), "pchip")
    return out


def plan_emd_geometry(P) -> List[Case]:
    """One decomposition, many launch geometries: the kernel's result must not depend on how many thread blocks share a
    signal (multi-SM mode with a software group barrier) -- every case is compared with PyEMD bit for bit."""
    out: List[Case] = []

    def add(label, N, sig, emd=None, G=1, tier=1):
        emd = dict(emd or {})
        out.append(Case(f"emd-geometry/{label}/{sig}@{N}", "emd-geometry", "emd", N, sig, SEED, emd=emd,
                        gpu_extra=dict(blocks_per_signal=G), checks=_emd_checks(emd), tier=tier,
                        note=f"blocks_per_signal={G}"))

    Ns = list(P.get("geometry_N", [5000]))
    for N in Ns:
        for sig in ("noise", "chirp_am"):
            for G in (1, 2, 4, 8):
                add(f"G={G}", N, sig, G=G, tier=1 if G in (1, 4) else 2)
    Nm = Ns[min(1, len(Ns) - 1)]
    for sp, det in (("akima", "parabol"), ("pchip", "simple"), ("quadratic", "parabol"), ("cubic_hermite", "simple"),
                    ("linear", "parabol"), ("slinear", "simple"), ("cubic", "parabol")):
        add(f"{sp}+{det},G=4", Nm, "noise", emd=dict(spline_kind=sp, extrema_detection=det), G=4, tier=2)
    add("FIXE_H=3,G=4", Nm, "noise", emd=dict(FIXE_H=3), G=4, tier=2)
    add("FIXE=5,G=3", Nm, "chirp_am", emd=dict(FIXE=5), G=3, tier=2)
    add("nbsym=4,G=3", Nm, "noise", emd=dict(nbsym=4), G=3, tier=2)
    return out


def plan_emd_fuzz(P) -> List[Case]:
    out = []
    for i in range(P["fuzz"]):
        cfg = fuzz_config(i)
        out.append(Case(f"emd-fuzz/{i:04d}", "emd-fuzz", "emd", fuzz_N(i), f"fuzz:{i}", SEED, emd=cfg["emd"], T=cfg["T"],
                        max_imf=cfg["max_imf"], checks=_emd_checks(cfg["emd"]) if cfg["max_imf"] == -1 else ("recon",),
                        tier=1, note=f"cfg={jdump(cfg)}"))
    return out


def build_plan(P) -> List[Case]:
    plan: List[Case] = []
    plan += plan_emd_scale(P)
    plan += plan_ens_scale(P, "eemd") + plan_ens_scale(P, "ceemdan")
    plan += plan_trials(P)
    plan += plan_emd_branch(P) + plan_ens_branch(P, "eemd") + plan_ens_branch(P, "ceemdan")
    plan += plan_emd_edge(P) + plan_emd_fuzz(P) + plan_emd_geometry(P)
    plan = [c for c in plan if c.tier <= P["max_tier"]]
    ids = [c.id for c in plan]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate case ids: {sorted(dup)[:5]}"
    return plan


CPU_RANK = ["emd-edge", "emd-fuzz", "emd-geometry", "emd-branch", "eemd-branch", "ceemdan-branch", "trials", "emd-scale", "eemd-scale",
            "ceemdan-scale"]
GPU_RANK = ["emd-scale", "eemd-scale", "ceemdan-scale", "trials", "emd-geometry", "emd-branch", "eemd-branch", "ceemdan-branch",
            "emd-edge", "emd-fuzz"]   # edge/fuzz last: they are the most likely to trip a sticky CUDA error


def ordered(plan: List[Case], rank: List[str]) -> List[Case]:
    pos = {s: i for i, s in enumerate(rank)}
    return sorted(plan, key=lambda c: (pos.get(c.suite, 99), c.N if c.suite.endswith("scale") else 0))


# ======================================================================================================================
# PyEMD side: the reference, with identical noise injected
# ======================================================================================================================
def inj_noise_row(seed: int, trial: int, scale: float, N: int, kind: str) -> np.ndarray:
    """Noise of trial `trial`: a pure function of (seed, trial) so that PyEMD's worker processes and the GPU host code
    regenerate bit-identical rows without sharing state."""
    g = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(trial)]))
    if kind == "normal":
        return g.normal(0.0, scale, N)
    if kind == "uniform":
        return g.uniform(-scale / 2.0, scale / 2.0, N)
    raise ValueError(kind)


def inj_noise_matrix(seed: int, trials: int, scale: float, N: int, kind: str) -> np.ndarray:
    return np.stack([inj_noise_row(seed, i, scale, N, kind) for i in range(int(trials))])


class RefEMD(_PyEMD):
    """PyEMD's EMD, plus a count of sift iterations per IMF (spline fits between two end_condition() calls)."""

    def emd(self, S, T=None, max_imf=-1):
        self.sift_counts, self._sc = [], 0
        return super().emd(S, T, max_imf)

    def extract_max_min_spline(self, T, S):
        self._sc += 1
        return super().extract_max_min_spline(T, S)

    def end_condition(self, S, IMF):
        self.sift_counts.append(self._sc)
        self._sc = 0
        return super().end_condition(S, IMF)


class RefEEMD(_PyEEMD):
    """PyEMD's EEMD with ONE change: trial i's noise comes from inj_noise_row(seed, i, ...) instead of self.random.
    (This is PyEMD's own _trial_update, line for line, with the noise source swapped; it also works with parallel=True.)"""

    def set_injection(self, seed, kind):
        self._inj_seed, self._inj_kind = int(seed), kind

    def _trial_update(self, trial):
        noise = inj_noise_row(self._inj_seed, trial, self._scale, self._N, self._inj_kind)
        imfs = self.emd(self._S + noise, self._T, self.max_imf)
        trend = None
        if self.separate_trends:
            imfs, trend = self.EMD.get_imfs_and_trend()
        return (imfs, trend)


class RefCEEMDAN(_PyCEEMDAN):
    """PyEMD's CEEMDAN with ONE change: the (trials, N) noise block it draws once per call is the injected one."""

    def set_injection(self, seed, kind):
        self._inj_seed, self._inj_kind = int(seed), kind

    def generate_noise(self, scale, size):
        if isinstance(size, tuple) and len(size) == 2:
            return inj_noise_matrix(self._inj_seed, size[0], scale, size[1], self._inj_kind)
        return super().generate_noise(scale, size)

    def _eemd(self, S, T=None, max_imf=-1, progress=True):
        """PyEMD 1.10's CEEMDAN._eemd, line for line, except Pool.imap instead of Pool.imap_unordered: the trials are
        added in TRIAL order (deterministic, = PyEMD's parallel=False result) instead of completion order."""
        from multiprocessing import Pool
        if T is None:
            T = np.arange(len(S), dtype=S.dtype)
        self._S = S
        self._T = T
        self._N = N = len(S)
        self.max_imf = max_imf
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.imap
        else:
            map_pool = map
        self.E_IMF = np.zeros((1, N))
        for IMFs in map_pool(self._trial_update, range(self.trials)):
            if self.E_IMF.shape[0] < IMFs.shape[0]:
                num_new_layers = IMFs.shape[0] - self.E_IMF.shape[0]
                self.E_IMF = np.vstack((self.E_IMF, np.zeros(shape=(num_new_layers, N))))
            self.E_IMF[: IMFs.shape[0]] += IMFs
        if self.parallel:
            pool.close()
        return self.E_IMF / self.trials


def make_ref_object(c: Case, parallel: bool):
    if c.algo == "emd":
        return RefEMD(**c.emd)
    ens = dict(c.ens)
    trials = int(ens.pop("trials"))
    if c.algo == "eemd":
        width = float(ens.pop("noise_width", 0.05))
        kw: Dict[str, Any] = dict(trials=trials, noise_width=width, parallel=parallel, **ens)
        obj = RefEEMD(ext_EMD=_PyEMD(**c.emd), **kw) if c.ext_emd else RefEEMD(**kw, **c.emd)
    else:
        eps = float(ens.pop("epsilon", 0.005))
        kw = dict(trials=trials, epsilon=eps, parallel=parallel, **ens)
        obj = RefCEEMDAN(ext_EMD=_PyEMD(**c.emd), **kw) if c.ext_emd else RefCEEMDAN(**kw, **c.emd)
    obj.set_injection(c.inj_seed, c.inj_kind)
    return obj


def injected_noise(c: Case, S: np.ndarray) -> np.ndarray:
    """The (trials, N) noise block both libraries receive, scaled exactly as PyEMD scales its own noise."""
    trials = int(c.ens["trials"])
    if c.algo == "eemd":
        scale = float(c.ens.get("noise_width", 0.05)) * np.abs(np.max(S) - np.min(S))   # PyEMD EEMD.eemd(), verbatim
    else:
        scale = float(c.ens.get("noise_scale", 1.0))                                    # PyEMD CEEMDAN.__init__
    return inj_noise_matrix(c.inj_seed, trials, scale, S.size, c.inj_kind)


def ref_spec(c: Case, parallel: bool) -> Dict[str, Any]:
    spec = dict(v=HARNESS_REF_VERSION, pyemd=pyemd_version(), algo=c.algo, N=c.N, signal=c.signal, sig_seed=c.sig_seed,
                emd=c.emd, ens=c.ens, max_imf=c.max_imf, T=c.T, inj_seed=c.inj_seed, inj_kind=c.inj_kind,
                ext_emd=c.ext_emd, parallel=bool(parallel) if c.algo != "emd" else False)
    if c.algo == "ceemdan":
        spec["sum_order"] = CEEMDAN_REF_ORDER
    return spec


def compute_reference(c: Case, parallel: bool, max_reps: int) -> Dict[str, Any]:
    """Run PyEMD on the case. Returns {ok, error, times, out, counts, sift, n_real}."""
    S = make_signal(c.signal, c.N, c.sig_seed)
    T = make_T(c.T, c.N, c.sig_seed)
    res: Dict[str, Any] = dict(ok=False, error=None, times=[], out=None, counts=None, sift=None, n_real=None)

    def one():
        obj = make_ref_object(c, parallel)
        t0 = time.perf_counter()
        if c.algo == "emd":
            out = obj.emd(S, T, c.max_imf)
        elif c.algo == "eemd":
            out = obj.eemd(S, T=T, max_imf=c.max_imf)
        else:
            out = obj.ceemdan(S, T=T, max_imf=c.max_imf)
        return obj, np.array(out, dtype=np.float64), time.perf_counter() - t0

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            obj, out, dt = one()
        except Exception as e:      # PyEMD itself rejects this input/config: nothing to compare against
            res["error"] = f"{type(e).__name__}: {e}"
            return res
        res.update(ok=True, out=out, times=[dt])
        if c.algo == "emd":
            res["sift"] = [int(v) for v in obj.sift_counts]
            res["n_real"] = int(len(obj.imfs))
        elif c.algo == "eemd":
            res["counts"] = [int(v) for v in obj.ensemble_count()]
        if c.timing and max_reps > 1:      # cheap runs get a few repeats; a 10-minute run is measured once
            reps = 1 if dt > 20 else (2 if dt > 3 else max_reps)
            for _ in range(reps - 1):
                try:
                    res["times"].append(one()[2])
                except Exception:
                    break
    return res


# ----------------------------------------------------------------------------------------------------------------------
# on-disk cache of references
# ----------------------------------------------------------------------------------------------------------------------
class RefCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root, self.enabled = Path(root), enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(spec) -> str:
        return hashlib.sha1(jdump(spec).encode()).hexdigest()[:20]

    def exists(self, spec) -> bool:
        return self.enabled and (self.root / f"{self.key(spec)}.json").exists()

    def load(self, spec) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        k = self.key(spec)
        jp = self.root / f"{k}.json"
        if not jp.exists():
            return None
        try:
            res = json.loads(jp.read_text())
            if res.get("ok"):
                with np.load(self.root / f"{k}.npz") as z:
                    res["out"] = z["out"]
            return res
        except Exception:
            return None

    def store(self, spec, res) -> None:
        if not self.enabled:
            return
        k = self.key(spec)
        meta = {kk: vv for kk, vv in res.items() if kk != "out"}
        meta["spec"] = spec
        meta["host"] = dict(cpu=cpu_model(), threads=cpu_usable())
        if res.get("ok"):
            tmp = self.root / f"{k}.tmp.npz"
            np.savez(tmp, out=res["out"])
            os.replace(tmp, self.root / f"{k}.npz")
        (self.root / f"{k}.json").write_text(jdump(clean(meta)))


# ======================================================================================================================
# Phase 1: CPU references
# ======================================================================================================================
def _hist_key(c: Case):
    return (c.algo, c.signal, jdump(c.emd), jdump(c.ens), c.max_imf, c.T)


def skip_reason(c: Case, hist, args) -> Optional[str]:
    if args.ref_max_n and c.N > args.ref_max_n:
        return f"N={c.N} > --ref-max-n {args.ref_max_n}"
    if c.ref_policy != "budget":
        return None
    h = hist.get(_hist_key(c))
    if not h:
        return None                     # nothing smaller measured yet: run it, it is the first row
    n0, t0 = h[-1]
    est = t0 * (c.N / n0) * (math.log2(c.N) / max(math.log2(n0), 1.0)) if c.N > n0 else t0
    if est > args.cpu_budget:
        return f"predicted PyEMD time {fmt_s(est)} (from N={n0}: {fmt_s(t0)}) > --cpu-budget {fmt_s(args.cpu_budget)}"
    return None


def phase_reference(plan: List[Case], args, cache: RefCache, P) -> Dict[str, str]:
    """Compute (or load) the PyEMD reference of every case that wants one. Returns {case id: reason it was skipped}."""
    skipped: Dict[str, str] = {}
    hist: Dict[Any, List[Tuple[int, float]]] = {}
    todo = [c for c in ordered(plan, CPU_RANK) if c.ref_policy != "never"]
    log(f"\n=== CPU phase: PyEMD references for {len(todo)} cases "
        f"({cpu_usable()} CPU threads, cache: {cache.root if cache.enabled else 'off'}) ===")
    t_start = time.perf_counter()
    for i, c in enumerate(todo, 1):
        spec = ref_spec(c, True)
        res = cache.load(spec)
        src = "cached"
        if res is None:
            why = skip_reason(c, hist, args)
            if why:
                skipped[c.id] = why
                log(f"[{i:4d}/{len(todo)}] SKIP  {c.id:58s} {why}")
                continue
            t0 = time.perf_counter()
            res = compute_reference(c, True, P["repeats"])
            cache.store(spec, res)
            src = f"ran {fmt_s(time.perf_counter() - t0)}"
        if res.get("ok"):
            t = median(res["times"])
            hist.setdefault(_hist_key(c), []).append((c.N, t))
            log(f"[{i:4d}/{len(todo)}] ok    {c.id:58s} rows {res['out'].shape[0]:3d}  pyemd {fmt_s(t):>9s}  ({src})")
        else:
            log(f"[{i:4d}/{len(todo)}] ERR   {c.id:58s} PyEMD raised: {res['error']}  ({src})")
        # optional single-core baseline (EEMD / CEEMDAN only; EMD has no parallel mode)
        if args.serial_baseline and c.algo != "emd" and c.timing and res.get("ok") and c.N <= args.serial_max_n:
            sspec = ref_spec(c, False)
            sres = cache.load(sspec)
            if sres is None:
                est = median(res["times"]) * min(cpu_usable(), int(c.trials or 1))
                if est > args.cpu_budget:
                    log(f"           serial baseline skipped (predicted {fmt_s(est)})")
                    continue
                t0 = time.perf_counter()
                sres = compute_reference(c, False, 1)
                cache.store(sspec, sres)
                log(f"           serial baseline {fmt_s(median(sres['times']) if sres.get('ok') else None)}")
    log(f"=== CPU phase done in {fmt_s(time.perf_counter() - t_start)} ===")
    return skipped


def selfcheck_injection() -> None:
    """Prove that the noise injection really takes effect in PyEMD (parallel mode included): otherwise every EEMD /
    CEEMDAN comparison below would be comparing different random numbers. Raises AssertionError with a diagnosis."""
    log("\n=== harness self-check: noise injection into PyEMD ===")
    S = make_signal("noisy_mix", 240, 0)

    def run(algo, seed, **ens):
        c = Case("selfcheck", "selfcheck", algo, 240, "noisy_mix", 0, ens=_ens_params(algo, 6, **ens), inj_seed=seed)
        r = compute_reference(c, True, 1)
        assert r["ok"], f"PyEMD {algo} raised in the self-check: {r['error']}"
        return r["out"]

    base = RefEMD().emd(S)
    z = run("eemd", 1, noise_width=0.0)
    assert z.shape == base.shape and np.abs(z - base).max() < 1e-12, \
        "EEMD with noise_width=0 must reproduce plain EMD; the EEMD noise hook is not wired up as expected"
    a, b, d = run("eemd", 5), run("eemd", 5), run("eemd", 6)
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-12, "EEMD: same injection seed gave different results"
    assert a.shape != d.shape or np.abs(a - d).max() > 1e-9, "EEMD: different injection seed gave identical results"
    a, b, d = run("ceemdan", 5), run("ceemdan", 5), run("ceemdan", 6)
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-12, \
        "CEEMDAN: same injection seed gave different results (the noise hook is not effective)"
    assert a.shape != d.shape or np.abs(a - d).max() > 1e-9, "CEEMDAN: different injection seed gave identical results"
    log("injection OK (zero-noise EEMD == EMD; same seed reproducible; different seed differs)")


# ======================================================================================================================
# GPU side
# ======================================================================================================================
def load_gpu_libs(lib_dir: Optional[str]) -> None:
    global cp, GPUBatchEMD, GPUBatchEEMD, GPUBatchCEEMDAN, EMDOverflowWarning, GpuEMD, GpuEEMD, GpuCEEMDAN, gpu_kernels
    for d in (str(HERE), lib_dir):                    # lib_dir (if given) ends up first on sys.path
        if d and d not in sys.path:
            sys.path.insert(0, d)
    import cupy as _cp
    from ceemdan import GPUBatchCEEMDAN as _C
    from eemd import GPUBatchEEMD as _EE
    from emd import EMDOverflowWarning as _W, GPUBatchEMD as _E
    cp, GPUBatchEMD, GPUBatchEEMD, GPUBatchCEEMDAN, EMDOverflowWarning = _cp, _E, _EE, _C, _W
    try:                                              # the PyEMD drop-in classes (v3+); older libraries lack them
        from emd import EMD as _DE
        from eemd import EEMD as _DEE
        from ceemdan import CEEMDAN as _DC
        import kernels as _K
        GpuEMD, GpuEEMD, GpuCEEMDAN, gpu_kernels = _DE, _DEE, _DC, _K
    except Exception:
        GpuEMD = GpuEEMD = GpuCEEMDAN = gpu_kernels = None


def sync() -> None:
    cp.cuda.Device().synchronize()


def cuda_alive() -> bool:
    """False after a sticky CUDA error (illegal address, ...): every later call would fail for that reason alone."""
    try:
        float(cp.arange(8, dtype=cp.float64).sum())
        sync()
        return True
    except Exception:
        return False


def make_gpu_object(c: Case):
    if c.algo == "emd":
        return GPUBatchEMD(**c.emd, **c.gpu_extra)
    ens = dict(c.ens)
    trials = int(ens.pop("trials"))
    gx = dict(c.gpu_extra)
    if c.algo == "eemd":
        width = float(ens.pop("noise_width", 0.05))
        kw = dict(trials=trials, noise_width=width, **ens, **gx)
        return GPUBatchEEMD(ext_EMD=GPUBatchEMD(**c.emd), **kw) if c.ext_emd else GPUBatchEEMD(**kw, **c.emd)
    eps = float(ens.pop("epsilon", 0.005))
    kw = dict(trials=trials, epsilon=eps, **ens, **gx)
    return GPUBatchCEEMDAN(ext_EMD=GPUBatchEMD(**c.emd), **kw) if c.ext_emd else GPUBatchCEEMDAN(**kw, **c.emd)


def n_repeats(t_warm: float, max_reps: int) -> int:
    """TOTAL number of timed calls, decided from the first warm call: many for cheap calls, few for minute-long ones."""
    if t_warm > 60:
        return 1
    if t_warm > 15:
        return min(max_reps, 2)
    if t_warm > 3:
        return min(max_reps, 3)
    return max_reps


def run_gpu(c: Case, S: np.ndarray, T: Optional[np.ndarray], P) -> Dict[str, Any]:
    """Run the GPU library through its PyEMD-compatible method. First call = warm-up (compile / memory pool; also the
    result that is compared); timing cases then repeat it a few times, each bracketed by device synchronisation."""
    obj = make_gpu_object(c)
    noise = cp.asarray(injected_noise(c, S)) if c.algo != "emd" else None

    def call():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if c.algo == "emd":
                out = obj.emd(S, T, c.max_imf)
            elif c.algo == "eemd":
                out = obj.eemd(S, T, c.max_imf, False, noise)
            else:
                out = obj.ceemdan(S, T, c.max_imf, False, noise)
        return np.asarray(out), any(issubclass(x.category, EMDOverflowWarning) for x in w)

    sync()
    t0 = time.perf_counter()
    out, flagged = call()
    sync()
    cold = time.perf_counter() - t0
    times: List[float] = []
    deterministic: Optional[bool] = None
    if c.timing and cold <= 120:                 # a multi-minute call is measured once (the warm-up call itself)
        deterministic, total = True, None
        while total is None or len(times) < total:
            sync()
            t0 = time.perf_counter()
            o, f = call()
            sync()
            times.append(time.perf_counter() - t0)
            flagged = flagged or f
            deterministic = deterministic and bool(np.array_equal(out, o))
            del o
            if total is None:                    # the first warm call (not the compile-inflated cold one) sets the count
                total = n_repeats(times[0], P["repeats"])
    res: Dict[str, Any] = dict(out=out, flagged=flagged, cold=cold, times=times, deterministic=deterministic)
    if c.algo == "emd":
        n_real = int(cp.asnumpy(obj.last_imf_counts)[0])
        res.update(n_real=n_real, debug_n=[int(v) for v in cp.asnumpy(obj.last_debug[0])[0, :n_real]],
                   imfs=obj.imfs, max_iteration=int(obj.MAX_ITERATION))
    elif c.algo == "eemd":
        res["counts"] = [int(v) for v in obj.ensemble_count()]
    if flagged and c.algo != "emd":          # how many inner EMD runs were flagged (the drop-in methods only warn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = obj(S, T, c.max_imf, False, noise)
        res["n_overflow"] = int(r[2] if c.algo == "eemd" else r[1])
    del obj, noise
    return res


# ----------------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------------
def compare_rows(ref: np.ndarray, gpu: np.ndarray, peak: float, tol: float) -> Dict[str, Any]:
    """Row-by-row error. rel_k = max|gpu_k - ref_k| / max(max|ref_k|, 1e-5 * signal peak): relative to the ROW's own
    amplitude (so a small IMF is held to the same relative standard as a large one), with a floor so that rows far below
    the signal's rounding noise cannot dominate. rel_peak = worst abs error / signal peak, for reference.
    rms_err = sqrt(mean((gpu - ref)^2)) over all samples of all compared rows (report-only, like max_abs; inf if NaN)."""
    r, g = np.atleast_2d(ref), np.atleast_2d(gpu)
    nr, ng = r.shape[0], g.shape[0]
    m = min(nr, ng)
    floor = 1e-5 * max(peak, 1e-300)      # rows below this are held to an ABSOLUTE 1e-5*tol*peak (>> rounding noise)
    per_abs, per_rel = [], []
    sq_sum, n_cmp, nan_seen = 0.0, 0, False      # accumulators for the report-only RMS error
    for k in range(m):
        d = np.abs(g[k] - r[k])
        a = float("inf") if np.isnan(d).any() else (float(d.max()) if d.size else 0.0)
        sc = max(float(np.abs(r[k]).max()) if r.shape[1] else 0.0, floor)
        per_abs.append(a)
        per_rel.append(a / sc)
        nan_seen = nan_seen or bool(np.isnan(d).any())
        sq_sum += float(np.dot(d, d))
        n_cmp += int(d.size)
    first_bad_num = next((k for k, v in enumerate(per_rel) if v > tol), None)    # first row that differs NUMERICALLY
    first_bad = first_bad_num if first_bad_num is not None else (m if nr != ng else None)
    mx = max(per_abs) if per_abs else 0.0
    rms = float("inf") if nan_seen else (math.sqrt(sq_sum / n_cmp) if n_cmp else 0.0)
    return dict(rows_ref=nr, rows_gpu=ng, rows_equal=(nr == ng), max_abs=mx, rms_err=rms, rel=max(per_rel) if per_rel else 0.0,
                rel_peak=(mx / peak if peak > 0 else mx), first_bad=first_bad, first_bad_num=first_bad_num, per_rel=per_rel)


def imf_property_violations(imfs: np.ndarray, skip=()) -> List[int]:
    """PyEMD's own IMF test, applied to the GPU's output: |#extrema - #zero-crossings| <= 1. Holds for every IMF the
    default stopping rule accepts (it is half of that rule), so it needs no reference."""
    ex = _PyEMD()
    T = np.arange(imfs.shape[1], dtype=np.float64)
    bad = []
    with np.errstate(all="ignore"):
        for k in range(imfs.shape[0]):
            if k in skip:
                continue
            e = ex.find_extrema(T, imfs[k])
            if abs((len(e[0]) + len(e[2])) - len(e[4])) >= 2:
                bad.append(k)
    return bad


def sift_agreement(ref_sift: List[int], gpu_n: List[int], max_it: int) -> Tuple[int, int, Optional[int]]:
    """Compare per-IMF sift-iteration counts. GPU debug_n = loop counter n at exit; PyEMD's counter = spline fits, which
    equals n for a converged IMF and n-1 for one that hit MAX_ITERATION (the check precedes the fit)."""
    cmp_ = eq = 0
    first = None
    for j in range(min(len(gpu_n), len(ref_sift))):
        g, r = gpu_n[j], ref_sift[j]
        if g < 0:
            continue
        cmp_ += 1
        if g == r or (g >= max_it and r >= max_it - 1):
            eq += 1
        elif first is None:
            first = j
    return cmp_, eq, first


def new_record(c: Case) -> Dict[str, Any]:
    keys = ("ref_status ref_note ref_error ref_time ref_times ref_serial_time gpu_time gpu_cold gpu_times speedup "
            "speedup_serial rows_ref rows_gpu rows_equal max_abs rel rel_peak first_bad per_rel counts_equal "
            "counts_varied sift_cmp sift_equal sift_first_diff flagged cap_hit n_overflow finite recon_err recon_tol "
            "imf_violations deterministic gpu_error tb amp_ratio rms_err").split()
    rec: Dict[str, Any] = {k: None for k in keys}
    rec.update(id=c.id, suite=c.suite, algo=c.algo, N=c.N, signal=c.signal, trials=c.trials, mode="", status="PASS",
               reasons=[], note=c.note, config=dataclasses.asdict(c))
    return rec


def run_case(c: Case, args, P, cache: RefCache, skipped: Dict[str, str]) -> Dict[str, Any]:
    rec = new_record(c)
    S = make_signal(c.signal, c.N, c.sig_seed)
    T = make_T(c.T, c.N, c.sig_seed)
    peak = float(np.max(np.abs(S))) if S.size else 0.0

    # ---- the reference -------------------------------------------------------------------------------------------
    ref = None
    if c.ref_policy == "never":
        rec.update(ref_status="none", mode="gpu-only")
    else:
        ref = cache.load(ref_spec(c, True))
        if ref is None:
            rec.update(ref_status="skipped", mode="gpu-only", ref_note=skipped.get(c.id, "no cached reference"))
        elif not ref.get("ok"):
            rec.update(ref_status="error", ref_error=ref.get("error"))
            if not args.probe_rejected:
                rec.update(status="SKIP", mode="n/a",
                           reasons=[f"PyEMD itself raised ({ref.get('error')}); nothing to compare"])
                return rec
            rec["mode"], ref = "gpu-only", None
        else:
            rec.update(ref_status="ok", mode="vs-pyemd", ref_time=median(ref["times"]), ref_times=ref["times"])
            if c.algo != "emd":
                sres = cache.load(ref_spec(c, False))
                if sres and sres.get("ok"):
                    rec["ref_serial_time"] = median(sres["times"])

    # ---- run the GPU library ---------------------------------------------------------------------------------------
    try:
        g = run_gpu(c, S, T, P)
    except FatalGPUError:
        raise
    except Exception as e:
        rec.update(status="FAIL", gpu_error=f"{type(e).__name__}: {e}", tb=traceback.format_exc(limit=8))
        rec["reasons"].append(f"GPU library raised {rec['gpu_error']}" + (" where PyEMD succeeded" if ref else ""))
        if not cuda_alive():
            err = FatalGPUError(f"CUDA context unusable after case {c.id}: {rec['gpu_error']}")
            err.rec = rec
            raise err
        _free_gpu(c)
        return rec

    out = g["out"]
    gt = g["times"] or [g["cold"]]
    rec.update(gpu_time=median(gt), gpu_cold=g["cold"], gpu_times=g["times"], deterministic=g["deterministic"],
               flagged=g["flagged"], n_overflow=g.get("n_overflow"), finite=bool(np.isfinite(out).all()),
               rows_gpu=int(out.shape[0]))
    fails: List[str] = []
    warns: List[str] = []
    if not rec["finite"]:
        fails.append("GPU output contains NaN/inf")

    # which IMFs (EMD) hit MAX_ITERATION: that is a property of the signal, not a capacity limit
    cap_first = None
    if c.algo == "emd" and g.get("debug_n") is not None:
        capped = [j for j, v in enumerate(g["debug_n"]) if v >= g["max_iteration"]]
        rec["cap_hit"] = bool(capped)
        cap_first = capped[0] if capped else None
    else:
        capped = []

    # ---- match against PyEMD -----------------------------------------------------------------------------------------
    if ref is not None:
        cmp_ = compare_rows(ref["out"], out, peak, args.tol)
        rec.update({k: cmp_[k] for k in ("rows_ref", "rows_gpu", "rows_equal", "max_abs", "rms_err", "rel", "rel_peak", "first_bad",
                                         "per_rel")})
        msgs = []
        if not cmp_["rows_equal"]:
            msgs.append(f"row count differs: PyEMD {cmp_['rows_ref']} vs GPU {cmp_['rows_gpu']}")
        if cmp_["rel"] > args.tol:
            msgs.append(f"max rel err {cmp_['rel']:.2e} > tol {args.tol:g} (first bad row {cmp_['first_bad']})")
        if msgs:
            # excusable only if the rows start to DIFFER NUMERICALLY at/after an IMF that never converged; a missing or
            # extra row with every common row identical is not a rounding-amplification symptom
            explained = (rec["flagged"] and cap_first is not None and cmp_["first_bad_num"] is not None
                         and cmp_["first_bad_num"] >= cap_first)
            if explained:
                warns.append("; ".join(msgs) + f" -- from IMF {cap_first} on, which hit MAX_ITERATION "
                             "(a non-converged sift amplifies rounding differences)")
            else:
                fails.extend(msgs)
        if c.algo == "eemd":
            rec["counts_equal"] = (g["counts"] == ref["counts"])
            rec["counts_varied"] = len(set(ref["counts"])) > 1
            if not rec["counts_equal"]:
                fails.append(f"ensemble_count differs: PyEMD {ref['counts']} vs GPU {g['counts']}")
        if c.algo == "emd" and ref.get("sift") is not None and g.get("debug_n") is not None:
            n_cmp, n_eq, first = sift_agreement(ref["sift"], g["debug_n"], g["max_iteration"])
            rec.update(sift_cmp=n_cmp, sift_equal=n_eq, sift_first_diff=first)

    # ---- checks that need no reference -------------------------------------------------------------------------------
    row_peak = float(np.max(np.abs(out))) if out.size else 0.0
    rec["amp_ratio"] = row_peak / peak if peak > 0 else None      # >> 1: the spline extrapolation blew up
    if "recon" in c.checks and out.ndim == 2:
        # sum(rows) - S is limited by cancellation at the size of the LARGEST row, not of the signal
        rec["recon_err"] = float(np.max(np.abs(S - out.sum(axis=0)))) if out.shape[0] else float(np.max(np.abs(S)))
        rec["recon_tol"] = (1e-8 if c.algo == "emd" else 1e-12) + 1e-11 * max(peak, row_peak) * max(out.shape[0], 1)
        if not rec["recon_err"] <= rec["recon_tol"]:
            fails.append(f"reconstruction error {rec['recon_err']:.2e} > {rec['recon_tol']:.2e}")
    if rec["amp_ratio"] is not None and rec["amp_ratio"] > 1e6:
        # PyEMD itself does this on some inputs (non-uniform T + parabol, near-degenerate envelopes); when the
        # output equals PyEMD's it is not the GPU library's doing, but it deserves to be visible either way
        (warns if ref is None else rec["reasons"]).append(
            f"decomposition diverged: an IMF reaches {rec['amp_ratio']:.1e}x the signal's peak"
            + ("" if ref is None else " (identical in PyEMD)"))
    if "imf_property" in c.checks and c.algo == "emd" and g.get("imfs") is not None:
        bad = imf_property_violations(g["imfs"], skip=set(capped))
        rec["imf_violations"] = len(bad)
        if bad:
            fails.append(f"{len(bad)} IMF(s) violate |#extrema - #zero-crossings| <= 1 (rows {bad[:6]})")

    # ---- overflow flags ------------------------------------------------------------------------------------------------
    if rec["flagged"] and not c.expect_overflow:
        if c.algo == "emd" and not rec["cap_hit"]:
            fails.append("overflow flag raised WITHOUT any IMF hitting MAX_ITERATION: the IMF slots ran out or an "
                         "envelope was degenerate -- this is the capacity problem, not a property of the signal")
        elif not msgs_fail_already(fails):
            warns.append("overflow flag raised: " + ("a sift hit MAX_ITERATION (PyEMD's own cap; the IMF is accepted as "
                                                      "it stands)" if rec["cap_hit"] else
                                                      "some inner EMD run hit MAX_ITERATION or ran out of slots"))

    info = [r for r in rec["reasons"] if r.startswith("decomposition diverged")]     # informational, never changes status
    rec["status"] = "FAIL" if fails else ("WARN" if warns else "PASS")
    rec["reasons"] = fails + warns + [r for r in info if r not in warns]
    if rec["ref_time"] and rec["gpu_time"]:
        rec["speedup"] = rec["ref_time"] / rec["gpu_time"]
        if rec["ref_serial_time"]:
            rec["speedup_serial"] = rec["ref_serial_time"] / rec["gpu_time"]
    del out, g
    _free_gpu(c)
    return rec


def msgs_fail_already(fails: List[str]) -> bool:
    return any("rel err" in f or "row count" in f or "ensemble_count" in f for f in fails)


def _free_gpu(c: Case) -> None:
    gc.collect()
    if c.N >= 16384 or (c.trials or 0) >= 200:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def case_line(rec: Dict[str, Any], i: int, n: int) -> str:
    parts = [f"[{i:4d}/{n}] {rec['status']:4s} {rec['id']:58s}"]
    if rec["rows_ref"] is not None:
        parts.append(f"rows {rec['rows_ref']}/{rec['rows_gpu']}  rel {fmt_e(rec['rel'])}  abs {fmt_e(rec['max_abs'])}")
    elif rec["gpu_time"] is not None:
        parts.append(f"[gpu-only] rows {rec['rows_gpu']}")
    if rec["sift_cmp"]:
        parts.append(f"sift {rec['sift_equal']}/{rec['sift_cmp']}")
    if rec["flagged"]:
        parts.append("FLAG")
    if rec["ref_time"] is not None and rec["gpu_time"] is not None and rec["suite"] in (
            "emd-scale", "eemd-scale", "ceemdan-scale", "trials"):
        parts.append(f"pyemd {fmt_s(rec['ref_time'])} gpu {fmt_s(rec['gpu_time'])} {fmt_x(rec['speedup'])}")
    elif rec["gpu_time"] is not None and rec["suite"] in ("emd-scale", "eemd-scale", "ceemdan-scale"):
        parts.append(f"gpu {fmt_s(rec['gpu_time'])}")
    line = "  ".join(parts)
    for r in rec["reasons"][:2]:
        line += f"\n           -> {r}"
    if rec["status"] == "SKIP" and rec["reasons"]:
        pass
    return line


# ======================================================================================================================
# Suite "stats": each library's OWN noise generator.  No exact comparison is possible there (different RNG streams),
# so: (1) the generator itself, (2) determinism / chunk-invariance, (3) a two-sample permutation energy test of
# "M PyEMD runs" vs "M GPU runs".  Under H0 (same distribution) p is uniform on (0,1]; the smallest p reachable with
# M = 6 per side is 2/924, so FAIL is p <= 0.003 (the observed split is the most extreme of all of them), WARN is p < 0.01.
# This is a GROSS-ERROR check (wrong noise scale/kind, wrong averaging, broken seeding): with M runs per side it cannot
# resolve small differences, and CEEMDAN's run-to-run spread is heavy-tailed. The exact, high-power comparison is the
# injected-noise one above.
# PyEMD is run with parallel=False here: its parallel EEMD pickles self.random into every task, so tasks can start
# from the same RNG state and reuse noise -- the serial path is the statistically clean baseline.
# ======================================================================================================================
class MemCache:
    """Stand-in for RefCache holding one in-memory reference (used by the smoke test)."""
    def __init__(self, ref):
        self.ref = ref

    def load(self, spec):
        return self.ref


def stat_record(sid: str, algo: str, status: str, reasons: List[str], **metrics) -> Dict[str, Any]:
    c = Case(sid, "stats", algo, 0, "noisy_mix")
    rec = new_record(c)
    rec.update(status=status, reasons=reasons, mode="statistical", metrics=metrics, N=metrics.get("N", 0))
    return rec


def energy_test(py: List[np.ndarray], gp: List[np.ndarray]) -> Dict[str, float]:
    runs = py + gp
    K = min(r.shape[0] for r in runs)
    runs = [r[:K] for r in runs]
    n, nA = len(runs), len(py)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.sqrt(np.sum((runs[i] - runs[j]) ** 2)))

    def stat(A):
        A = list(A)
        B = [i for i in range(n) if i not in A]
        return (2 * D[np.ix_(A, B)].mean() - D[np.ix_(A, A)].sum() / (len(A) * (len(A) - 1))
                - D[np.ix_(B, B)].sum() / (len(B) * (len(B) - 1)))

    obs = stat(range(nA))
    ge = tot = 0
    for comb in itertools.combinations(range(n), nA):
        tot += 1
        ge += stat(comb) >= obs - 1e-12 * abs(obs)
    iu = lambda idx: [D[i, j] for i, j in itertools.combinations(idx, 2)]
    return dict(p=ge / tot, energy=obs, rows_used=K, d_pyemd=float(np.mean(iu(range(nA)))),
                d_gpu=float(np.mean(iu(range(nA, n)))), d_cross=float(D[:nA, nA:].mean()))


def _gpu_native(algo, S, trials, kind, seed, **extra):
    if algo == "eemd":
        return GPUBatchEEMD(trials=trials, noise_width=0.05, noise_kind=kind, seed=seed, **extra).eemd(S)
    return GPUBatchCEEMDAN(trials=trials, epsilon=0.005, noise_kind=kind, seed=seed, **extra).ceemdan(S)


def _pyemd_native(algo, S, trials, kind, seed, cache: RefCache) -> np.ndarray:
    spec = dict(v=HARNESS_REF_VERSION, pyemd=pyemd_version(), stat="native", algo=algo, N=S.size, trials=trials,
                kind=kind, seed=seed, sig_seed=SEED)
    r = cache.load(spec)
    if r is None:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            if algo == "eemd":
                e = _PyEEMD(trials=trials, noise_width=0.05, parallel=False, noise_kind=kind)
                e.noise_seed(seed)
                out = e.eemd(S)
            else:
                out = _PyCEEMDAN(trials=trials, epsilon=0.005, parallel=False, noise_kind=kind, seed=seed).ceemdan(S)
        r = dict(ok=True, out=np.array(out), times=[time.perf_counter() - t0])
        cache.store(spec, r)
    return r["out"]


def run_stats(args, P, cache: RefCache) -> List[Dict[str, Any]]:
    M = int(P["stats_runs"])
    if M < 3:
        return []
    N = P["stats_N"]
    S = make_signal("noisy_mix", N, SEED)
    peak = float(np.max(np.abs(S)))
    recs: List[Dict[str, Any]] = []
    log(f"\n=== statistical suite (native noise, N={N}, {M} runs per library) ===")

    def emit(rec):
        recs.append(rec)
        log(f"  {rec['status']:4s} {rec['id']:58s} " + "; ".join(rec["reasons"][:1]))

    for algo in ("eemd", "ceemdan"):
        cls = GPUBatchEEMD if algo == "eemd" else GPUBatchCEEMDAN
        trials = int(P["stats_trials"][algo])
        # (1) the generator
        for kind in ("normal", "uniform"):
            try:
                x = cp.asnumpy(cls(trials=1, noise_kind=kind, seed=11).generate_noise(2.0, 400000))
                if kind == "normal":
                    ok = abs(x.mean()) < 5 * 2.0 / math.sqrt(x.size) and abs(x.std() / 2.0 - 1) < 0.01
                else:
                    ok = (x.min() >= -1.0 and x.max() <= 1.0 and abs(x.std() / (2.0 / math.sqrt(12)) - 1) < 0.01
                          and x.max() > 0.99 and x.min() < -0.99)
                emit(stat_record(f"stats/{algo}/generate_noise/{kind}", algo, "PASS" if ok else "FAIL",
                                 [] if ok else [f"generate_noise(2.0) moments off: mean {x.mean():.4f}, std {x.std():.4f}, "
                                                f"range [{x.min():.3f}, {x.max():.3f}]"], N=N))
            except Exception as e:
                emit(stat_record(f"stats/{algo}/generate_noise/{kind}", algo, "FAIL", [f"{type(e).__name__}: {e}"], N=N))
        # (2) determinism
        try:
            a = _gpu_native(algo, S, 12, "normal", 7)
            b = _gpu_native(algo, S, 12, "normal", 7)
            ch = _gpu_native(algo, S, 12, "normal", 7, chunk_trials=5)
            o = cls(trials=12, seed=7)
            r1 = o.eemd(S) if algo == "eemd" else o.ceemdan(S)
            o.noise_seed(7)
            r2 = o.eemd(S) if algo == "eemd" else o.ceemdan(S)
            same = lambda u, v: u.shape == v.shape and float(np.max(np.abs(u - v))) <= 1e-12 * max(peak, 1.0)
            probs = []
            if not same(a, b):
                probs.append("same seed -> different result")
            if not same(a, ch):
                probs.append("chunk_trials=5 changes the result for the same seed (noise must not depend on chunking)")
            if not same(r1, r2):
                probs.append("noise_seed(7) does not restore the stream")
            emit(stat_record(f"stats/{algo}/seed-determinism", algo, "FAIL" if probs else "PASS", probs, N=N))
        except Exception as e:
            emit(stat_record(f"stats/{algo}/seed-determinism", algo, "FAIL", [f"{type(e).__name__}: {e}"], N=N))
        # (3) the distribution of the result
        for kind in ("normal", "uniform"):
            sid = f"stats/{algo}/native-noise/{kind}"
            try:
                py = [_pyemd_native(algo, S, trials, kind, 1000 + i, cache) for i in range(M)]
                gp = [np.asarray(_gpu_native(algo, S, trials, kind, 2000 + i)) for i in range(M)]
                t = energy_test(py, gp)
                st = "FAIL" if t["p"] <= 0.003 else ("WARN" if t["p"] < 0.01 else "PASS")
                why = [] if st == "PASS" else [f"permutation energy test p={t['p']:.4f}: GPU and PyEMD results do not look "
                                               f"like samples of one distribution (mean pairwise distance PyEMD-PyEMD "
                                               f"{t['d_pyemd']:.3g}, GPU-GPU {t['d_gpu']:.3g}, cross {t['d_cross']:.3g})"]
                emit(stat_record(sid, algo, st, why, N=N, trials=trials, runs=M, noise_kind=kind, **t))
            except Exception as e:
                emit(stat_record(sid, algo, "FAIL", [f"{type(e).__name__}: {e}"], N=N))
    return recs


# ======================================================================================================================
# Suite "api": the PyEMD drop-in surface. Every check compares with PyEMD run IN THIS PROCESS with parallel=False
# (deterministic, no worker processes next to CUDA); sizes are small, the whole suite takes seconds.
# ======================================================================================================================
def api_record(sid: str, status: str, reasons: List[str], algo: str = "emd", **metrics) -> Dict[str, Any]:
    c = Case(sid, "api", algo, int(metrics.get("N", 0)), "api")
    rec = new_record(c)
    rec.update(status=status, reasons=reasons, mode="api", metrics=metrics, N=int(metrics.get("N", 0)))
    return rec


def _bits_equal(a, b) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _diff(name: str, ref, got, tol: float) -> Optional[str]:
    """None if bit-identical; otherwise a description (FAIL-worthy if beyond tol, else prefixed 'WARN:')."""
    ref, got = np.asarray(ref, dtype=np.float64), np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape:
        return f"{name}: shape {got.shape} != PyEMD {ref.shape}"
    if ref.tobytes() == got.tobytes():
        return None
    with np.errstate(all="ignore"):
        scale = max(float(np.nanmax(np.abs(ref))) if ref.size else 0.0, 1e-300)
        rel = float(np.nanmax(np.abs(ref - got))) / scale if ref.size else 0.0
    pre = "WARN: " if rel <= tol else ""
    return f"{pre}{name}: max rel diff {rel:.3g} (not bit-identical)"


def run_api(args, P) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    log("\n=== api suite (PyEMD drop-in surface, compared with PyEMD parallel=False in-process) ===")
    if GpuEMD is None:
        rec = api_record("api/import", "FAIL", ["the library has no drop-in classes EMD / EEMD / CEEMDAN (emd.py, eemd.py, "
                                                "ceemdan.py) or no kernels module"])
        log(f"  FAIL {rec['id']:58s} " + rec["reasons"][0])
        return [rec]
    tol = float(args.tol)

    def emit(sid, problems, algo="emd", **m):
        hard = [p for p in problems if not p.startswith("WARN:")]
        st = "FAIL" if hard else ("WARN" if problems else "PASS")
        rec = api_record(sid, st, problems, algo=algo, **m)
        recs.append(rec)
        log(f"  {st:4s} {sid:58s} " + "; ".join(problems[:2]))

    def check(sid, fn, algo="emd", **m):
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                probs = [p for p in fn() if p]
        except Exception as e:
            probs = [f"{type(e).__name__}: {e}"]
        emit(sid, probs, algo=algo, **m)

    rng = np.random.RandomState(1234 + SEED)
    N = 600
    t = np.arange(N, dtype=np.float64)
    S = np.sin(t / 7.0) + 0.4 * np.sin(t / 2.3) + 0.1 * rng.randn(N)

    # ---- EMD drop-in -------------------------------------------------------------------------------------------------
    def emd_call():
        out = []
        for kw in ({}, dict(spline_kind="akima"), dict(extrema_detection="parabol"), dict(FIXE=4), dict(FIXE_H=2),
                   dict(nbsym=3, spline_kind="pchip")):
            ref = _PyEMD(**kw)(S)
            g = GpuEMD(**kw)
            got = g(S)
            out.append(_diff(f"EMD({kw})(S)", ref, got, tol))
            Tm = np.linspace(0, 3, N)
            out.append(_diff(f"EMD({kw}).emd(S, T, max_imf=3)", _PyEMD(**kw).emd(S, Tm, max_imf=3),
                             GpuEMD(**kw).emd(S, Tm, max_imf=3), tol))
            p = _PyEMD(**kw)
            p.emd(S)
            out.append(_diff(f"get_imfs_and_trend imfs {kw}", p.get_imfs_and_trend()[0], g.get_imfs_and_trend()[0], tol))
            out.append(_diff(f"get_imfs_and_trend trend {kw}", p.get_imfs_and_trend()[1], g.get_imfs_and_trend()[1], tol))
            out.append(_diff(f"get_imfs_and_residue {kw}", p.get_imfs_and_residue()[1], g.get_imfs_and_residue()[1], tol))
        return out
    check("api/EMD()(S), emd(S,T,max_imf), get_imfs_and_*", emd_call, N=N)

    def emd_2d():
        X = np.stack([S, S[::-1].copy(), rng.randn(N), np.cumsum(rng.randn(N))])
        got = GpuEMD()(X)
        if not isinstance(got, list) or len(got) != len(X):
            return [f"EMD()(2-D) returned {type(got).__name__} of length {len(got) if hasattr(got, '__len__') else '?'}"]
        return [_diff(f"row {i}", _PyEMD().emd(X[i]), got[i], tol) for i in range(len(X))]
    check("api/EMD()(2-D array) = PyEMD row by row", emd_2d, N=N)

    def helpers():
        g, p = GpuEMD(), _PyEMD()
        out = []
        for a, b in zip(p.find_extrema(t, S), g.find_extrema(t, S)):
            out.append(_diff("find_extrema", a, b, tol))
        return out
    check("api/PyEMD helper methods (find_extrema) on the drop-in", helpers, N=N)

    # ---- batch call --------------------------------------------------------------------------------------------------
    def batch_mixed():
        sigs = [S, rng.randn(N), np.cumsum(rng.randn(N)), np.sin(t / 3.0), np.sin(t / 50.0) + 0.01 * t,
                rng.rand(N) * 3, np.sign(np.sin(t / 9.0)) + 0.05 * rng.randn(N), S * 1e6, S * 1e-6]
        X = np.stack(sigs)
        g = GPUBatchEMD()
        imfs, n, err = g(cp.asarray(X))[:3]
        n, err = cp.asnumpy(n), cp.asnumpy(err)
        out = []
        if int(np.count_nonzero(err)):
            out.append(f"batch error codes {err.tolist()}")
        if imfs.shape[1] != int(n.max()):
            out.append(f"batch output has {imfs.shape[1]} IMF slots, the largest decomposition needs {int(n.max())} "
                       "(the result is not trimmed)")
        h = cp.asnumpy(imfs)
        for i, x in enumerate(sigs):
            out.append(_diff(f"batch row {i}", _PyEMD().emd(x), h[i, :int(n[i])], tol))
        return out
    check("api/batch call (9 mixed signals, one launch)", batch_mixed, N=N)

    def growth():
        L = gpu_kernels.EMDKernelConfig().build()
        X = np.stack([rng.randn(300) for _ in range(5)] + [np.sin(np.arange(300) / 5.0)])
        imfs, n, err = L.run(cp.asarray(X), cp.asarray(np.arange(300.0)), max_imfs=2)[:3]
        n, err, h = cp.asnumpy(n), cp.asnumpy(err), cp.asnumpy(imfs)
        out = [f"errors {err.tolist()}"] if int(np.count_nonzero(err)) else []
        for b in range(len(X)):
            out.append(_diff(f"signal {b}", _PyEMD().emd(X[b]), h[b, :int(n[b])], tol))
        return out
    check("api/IMF-buffer growth (start with 2 slots)", growth, N=300)

    def no_overflow_warning():
        out = []
        for mi in (2, 5):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                got = GpuEMD(MAX_ITERATION=mi).emd(S)
            ow = [x for x in w if EMDOverflowWarning is not None and issubclass(x.category, EMDOverflowWarning)]
            if ow:
                out.append(f"MAX_ITERATION={mi}: an EMDOverflowWarning was issued ({ow[0].message})")
            out.append(_diff(f"MAX_ITERATION={mi}", _PyEMD(MAX_ITERATION=mi).emd(S), got, tol))
        return out
    check("api/MAX_ITERATION hit: PyEMD's result, no overflow warning", no_overflow_warning, N=N)

    def exceptions():
        cands = []                                    # (signal, T, kwargs, PyEMD's exception type)
        for seed in range(3000):
            r = np.random.RandomState(seed)
            n = int(r.randint(6, 30))
            x = np.round(r.randn(n) * 2) if seed % 3 == 0 else r.randn(n) * np.exp(r.randn(n))
            Tx = np.sort(r.rand(n)) * 10 if seed % 2 else None
            for mode, kind in (("parabol", "cubic"), ("parabol", "akima"), ("parabol", "pchip"), ("simple", "akima")):
                kw = dict(extrema_detection=mode, spline_kind=kind)
                try:
                    _PyEMD(**kw).emd(x.copy(), Tx)
                except Exception as ex:
                    cands.append((x, Tx, kw, type(ex)))
            if len(cands) >= 6:
                break
        x = rng.randn(200)
        Td = np.arange(200.0)
        Td[101] = Td[100]                             # repeated time stamp (PyEMD asserts unique times)
        Tl = np.arange(150.0)                         # wrong length
        for Tx, kw in ((Td, dict(extrema_detection="parabol")), (Tl, {})):
            try:
                _PyEMD(**kw).emd(x, Tx)
            except Exception as ex:
                cands.append((x, Tx, kw, type(ex)))
        out = []
        for x, Tx, kw, exn in cands:
            try:
                GpuEMD(**kw).emd(x, Tx)
                out.append(f"PyEMD raises {exn.__name__} ({kw}, N={len(x)}), the GPU returned a result")
            except Exception as ex:
                if not isinstance(ex, exn):
                    out.append(f"PyEMD raises {exn.__name__}, the GPU {type(ex).__name__} ({kw}, N={len(x)}): an "
                               f"`except {exn.__name__}` written for PyEMD would not catch it")
        if not cands:
            out.append("WARN: no PyEMD-raising input found (nothing to check)")
        return out
    check("api/exceptions where PyEMD raises", exceptions, N=40)

    def selftests():
        out = []
        for kind in ("cubic", "akima", "pchip", "cubic_hermite", "quadratic", "linear", "slinear"):
            for det in ("simple", "parabol"):
                r = GpuEMD(spline_kind=kind, extrema_detection=det).selftest(warn=False)
                if not r.get("ok", False):
                    out.append(f"selftest {kind}/{det}: " + "; ".join(r.get("mismatches", [])[:2]))
        return out
    check("api/selftest() of every spline x extrema kernel", selftests)

    def autotune():
        n2 = 20000
        x = np.sin(np.arange(n2) / 13.0) + 0.3 * np.random.RandomState(5).randn(n2)
        g = GPUBatchEMD()
        res = []
        for _ in range(3):
            imfs, n = g(cp.asarray(x[None, :]))[:2]
            res.append(cp.asnumpy(imfs[0, :int(n[0])]))
        out = [None if _bits_equal(res[0], r) else "repeated calls (auto-tuned launch shape) differ" for r in res[1:]]
        out.append(_diff("vs PyEMD", _PyEMD().emd(x), res[0], tol))
        return out
    check("api/auto-tuned launch shape: deterministic, = PyEMD (N=20000)", autotune, N=20000)

    # ---- EEMD / CEEMDAN with a seed (numpy noise stream, as PyEMD) ----------------------------------------------------
    def eemd_seeded():
        out = []
        for kw in ({}, dict(noise_kind="uniform"), dict(separate_trends=True), dict(spline_kind="akima")):
            p = _PyEEMD(trials=10, parallel=False, **kw)
            p.noise_seed(42)
            r1, r2 = p.eemd(S), p.eemd(S[::-1].copy())
            g = GpuEEMD(trials=10, **kw)
            g.noise_seed(42)
            q1, q2 = g.eemd(S), g.eemd(S[::-1].copy())
            out += [_diff(f"eemd {kw}", r1, q1, tol), _diff(f"second call {kw}", r2, q2, tol),
                    _diff(f"ensemble_std {kw}", p.ensemble_std(), g.ensemble_std(), tol),
                    _diff(f"ensemble_count {kw}", np.asarray(p.ensemble_count()), np.asarray(g.ensemble_count()), 0)]
            ka, kb = sorted(p.all_imfs), sorted(g.all_imfs)
            if ka != kb:
                out.append(f"all_imfs keys {kb} != PyEMD {ka}")
            else:
                out += [_diff(f"all_imfs[{k}] {kw}", np.asarray(p.all_imfs[k]), np.asarray(g.all_imfs[k]), tol) for k in ka]
            ip, rp = p.get_imfs_and_residue()
            ig, rg = g.get_imfs_and_residue()
            out += [_diff(f"residue {kw}", rp, rg, tol)]
        g = GpuEEMD(trials=10, seed=3)
        out.append(_diff("EEMD()(S) == eemd(S)", GpuEEMD(trials=10, seed=3).eemd(S), g(S), tol))
        return out
    check("api/EEMD seeded = PyEMD (same seed, numpy stream)", eemd_seeded, algo="eemd", N=N)

    def ceemdan_seeded():
        out = []
        for kw in ({}, dict(noise_kind="uniform"), dict(extrema_detection="parabol", spline_kind="akima"),
                   dict(noise_scale=2.0, beta_progress=False)):
            p = _PyCEEMDAN(trials=8, parallel=False, seed=7, **kw)
            r = p.ceemdan(S)
            g = GpuCEEMDAN(trials=8, seed=7, **kw)
            q = g.ceemdan(S)
            out += [_diff(f"ceemdan {kw}", r, q, tol), _diff(f"residue {kw}", p.residue, g.residue, tol)]
        p = _PyCEEMDAN(trials=8, parallel=False, seed=7)
        out.append(_diff("ceemdan(S, max_imf=2)", p.ceemdan(S, max_imf=2), GpuCEEMDAN(trials=8, seed=7).ceemdan(S, max_imf=2), tol))
        return out
    check("api/CEEMDAN seeded = PyEMD (same seed, numpy stream)", ceemdan_seeded, algo="ceemdan", N=N)

    def ext_emd():
        out = []
        cpu = _PyEMD(spline_kind="akima", extrema_detection="parabol", FIXE_H=2)
        p = _PyEEMD(trials=6, ext_EMD=cpu, parallel=False)
        p.noise_seed(5)
        g = GpuEEMD(trials=6, ext_EMD=_PyEMD(spline_kind="akima", extrema_detection="parabol", FIXE_H=2))
        g.noise_seed(5)
        out.append(_diff("EEMD(ext_EMD=PyEMD.EMD(...))", p.eemd(S), g.eemd(S), tol))
        p = _PyCEEMDAN(trials=6, ext_EMD=_PyEMD(spline_kind="pchip"), parallel=False, seed=9)
        g = GpuCEEMDAN(trials=6, ext_EMD=_PyEMD(spline_kind="pchip"), seed=9)
        out.append(_diff("CEEMDAN(ext_EMD=PyEMD.EMD(...))", p.ceemdan(S), g.ceemdan(S), tol))
        return out
    check("api/ext_EMD given as a CPU PyEMD.EMD", ext_emd, algo="eemd", N=N)
    return recs


# ======================================================================================================================
# Phase 2: GPU
# ======================================================================================================================
def _gpu_hist_key(algo, signal, emd, max_imf, T):
    return (algo, signal, jdump(emd), max_imf, T)


def predict_gpu_time(c: Case, gpu_hist) -> Optional[float]:
    """Predicted GPU wall time of a scale case from the finished smaller-N cases of the same family (time per trial is
    extrapolated with the local power-law exponent, clamped to [1, 2.5]; 1.3 if only one point). None = no history."""
    h = sorted(x for x in gpu_hist.get(_gpu_hist_key(c.algo, c.signal, c.emd, c.max_imf, c.T), []) if x[0] < c.N)
    if not h:
        return None
    n1, t1 = h[-1]
    b = 1.3
    if len(h) >= 2:
        n0, t0 = h[-2]
        if n1 > n0 and t0 > 0 and t1 > 0:
            b = min(max(math.log(t1 / t0) / math.log(n1 / n0), 1.0), 2.5)
    return t1 * (c.N / n1) ** b * (c.trials or 1)


def _heartbeat(label: str, interval: float = 120.0):
    """Background thread that logs a line every `interval` s while a long GPU call runs; set() the returned event to stop."""
    import threading
    stop, t0 = threading.Event(), time.perf_counter()

    def beat():
        while not stop.wait(interval):
            log(f"           ... still running {label} ({fmt_s(time.perf_counter() - t0)} elapsed)")

    threading.Thread(target=beat, daemon=True).start()
    return stop


def phase_gpu(plan: List[Case], args, P, cache, skipped, finish, records: List[Dict[str, Any]]) -> None:
    """Appends to `records` IN PLACE, so an interrupt or a fatal CUDA error cannot lose what was already measured."""
    todo = ordered(plan, GPU_RANK)
    safe = [c for c in todo if c.suite not in ("emd-edge", "emd-fuzz")]
    risky = [c for c in todo if c.suite in ("emd-edge", "emd-fuzz")]
    n = len(todo)
    log(f"\n=== GPU phase: {n} cases (tolerance {args.tol:g}) ===")
    missing = [c for c in todo if c.ref_policy != "never" and c.id not in skipped and not cache.exists(ref_spec(c, True))]
    if missing:
        log(f"NOTE: {len(missing)} case(s) have no cached PyEMD reference (cache: {cache.root if cache.enabled else 'off'}); "
            f"they run GPU-only with the reference-free checks. Run the CPU phase first for a full comparison "
            f"(a plain run without --phase does both).")
    i = 0
    done = {r["id"] for r in records}                    # cases carried over by --resume
    gpu_hist: Dict[Any, List[Tuple[int, float]]] = {}    # (N, seconds per trial) of finished scale cases
    for r in records:
        cfg = r.get("config") or {}
        if r.get("gpu_time") and str(cfg.get("suite", "")).endswith("-scale"):
            tr = (cfg.get("ens") or {}).get("trials") or 1
            gpu_hist.setdefault(_gpu_hist_key(cfg["algo"], cfg["signal"], cfg.get("emd", {}), cfg.get("max_imf", -1),
                                              cfg.get("T", "none")), []).append((cfg["N"], float(r["gpu_time"]) / tr))
    if done:
        log(f"resume: {len(done)} finished cases carried over, they are not re-run")

    def one(c):
        nonlocal i
        i += 1
        if c.id in done:
            return
        pr = None
        if c.suite.endswith("-scale") and c.timing:
            pr = predict_gpu_time(c, gpu_hist)
            if pr is not None and pr > args.gpu_budget:
                rec = new_record(c)
                rec.update(status="SKIP", mode="n/a", reasons=[
                    f"predicted GPU time {fmt_s(pr)} (extrapolated from smaller N) > --gpu-budget {fmt_s(args.gpu_budget)}; "
                    f"not run (raise --gpu-budget, or 0 for no limit, to run it)"])
                records.append(rec)
                log(case_line(rec, i, n))
                return
        log(f"[{i:4d}/{n}] ...... {c.id}  running" + (f" (predicted ~{fmt_s(pr)})" if pr is not None else ""))
        stop = _heartbeat(c.id)
        try:
            rec = run_case(c, args, P, cache, skipped)
        except FatalGPUError as e:
            if getattr(e, "rec", None) is not None:
                records.append(e.rec)
                log(case_line(e.rec, i, n))
            raise
        finally:
            stop.set()
        records.append(rec)
        log(case_line(rec, i, n))
        if rec.get("gpu_time") and c.suite.endswith("-scale"):
            gpu_hist.setdefault(_gpu_hist_key(c.algo, c.signal, c.emd, c.max_imf, c.T), []).append(
                (c.N, float(rec["gpu_time"]) / (c.trials or 1)))
        if (rec.get("gpu_time") or 0) > 30 or i % 5 == 0:     # keep results.json current so --resume always has something
            finish(records, partial=True)

    for c in safe:
        one(c)
    if args.want_stats and not any(r.get("suite") == "stats" for r in records):
        records.extend(run_stats(args, P, cache))
        finish(records, partial=True)
    if getattr(args, "want_api", False) and not any(r.get("suite") == "api" for r in records):
        records.extend(run_api(args, P))
        finish(records, partial=True)
    if risky:
        log("\n--- robustness suites (run last: a sticky CUDA error here cannot invalidate anything above) ---")
    for c in risky:
        one(c)


def gpu_smoke(args) -> int:
    """Child-process entry: a few tiny cases through the full machinery. Run BEFORE the long CPU phase so a wrong import,
    a changed API or a broken GPU is found in seconds, and in a separate process so CUDA stays uninitialised in the
    parent (PyEMD forks worker processes later). The three default-path cases are critical (exit code 1); the kernel
    VARIANTS are only reported (a broken variant is a finding for the branch suites, not a reason to stop)."""
    load_gpu_libs(args.lib_dir)
    log(f"GPU smoke test on {gpu_info().get('name')} ...")
    P = dict(PRESETS["smoke"])
    critical = [Case("smoke/emd", "smoke", "emd", 400, "noisy_mix", checks=("recon", "imf_property")),
                Case("smoke/eemd", "smoke", "eemd", 400, "noisy_mix", ens=_ens_params("eemd", 6)),
                Case("smoke/ceemdan", "smoke", "ceemdan", 400, "noisy_mix", ens=_ens_params("ceemdan", 4), checks=("recon",))]
    variants = [Case("smoke/emd[parabol]", "smoke", "emd", 400, "noisy_mix", emd=dict(extrema_detection="parabol")),
                Case("smoke/emd[akima]", "smoke", "emd", 400, "noisy_mix", emd=dict(spline_kind="akima"))]
    bad = 0
    for crit, cases in ((True, critical), (False, variants)):
        for c in cases:
            ref = compute_reference(c, False, 1)
            if not ref["ok"]:
                log(f"  ERR  {c.id}: PyEMD raised {ref['error']}")
                bad += crit
                continue
            try:
                rec = run_case(c, args, P, MemCache(ref), {})
            except FatalGPUError as e:
                log(f"  FAIL {c.id}: {e}")
                return 1
            log(case_line(rec, 0, 0).replace("[   0/0] ", "  "))
            if rec["status"] == "FAIL":
                if crit:
                    bad += 1
                else:
                    log("  NOTE: this kernel variant is broken; the branch suites will report it as FAIL. Continuing.")
    return 1 if bad else 0


# ======================================================================================================================
# Reporting
# ======================================================================================================================
SUITE_ORDER = ["emd-scale", "eemd-scale", "ceemdan-scale", "trials", "emd-geometry", "emd-branch", "eemd-branch",
               "ceemdan-branch", "stats", "api", "emd-edge", "emd-fuzz"]


def _suite_key(s: str) -> int:
    return SUITE_ORDER.index(s) if s in SUITE_ORDER else 99


def counts(recs) -> Dict[str, int]:
    out = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for r in recs:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def repro(rec, args) -> str:
    flags = f"--preset {args.preset} --phase gpu --tol {args.tol:g}"
    if args.seed:
        flags += f" --seed {args.seed}"
    if args.sizes:
        flags += " --sizes " + " ".join(str(n) for n in args.sizes)
    if args.trials:
        flags += f" --trials {args.trials}"
    if args.trials_sweep:
        flags += " --trials-sweep " + " ".join(str(n) for n in args.trials_sweep)
    if args.fuzz is not None:
        flags += f" --fuzz {args.fuzz}"
    return f'python {Path(sys.argv[0]).name} {flags} --only "{rec["id"]}"'


def methodology_block(args) -> str:
    """Short description of what this harness actually computes (kept next to the code it describes; update both together)."""
    return "\n".join([
        "## Methodology\n",
        "* **`rel` (the only quantity that decides PASS/FAIL on accuracy).** For every row k of the output matrix: "
        "rel_k = max_t |GPU_k(t) - PyEMD_k(t)| / max( max_t |PyEMD_k(t)|, 1e-5 * max_t |signal(t)| ). "
        "It is normalised by the peak amplitude of the PyEMD row itself, floored at 1e-5 times the peak of the input signal "
        "(so rows far below the signal's rounding noise cannot dominate). The case's `rel` is the maximum over rows.",
        "* **`max abs err` and `rms err` (report-only; they never enter PASS/FAIL).** `max abs err` = max over all compared rows and "
        "samples of |GPU - PyEMD|. `rms err` = sqrt(mean((GPU - PyEMD)^2)) over the same samples. Neither is normalised; "
        "both are inf if the difference contains NaN. Both are stored per case in results.json (`max_abs`, `rms_err`).",
        "* **Row matching.** Purely by position: row k of the GPU output is compared with row k of the PyEMD output (no sorting, "
        "alignment or search). If the row counts differ, only the first min(rows) rows are compared and the count mismatch is a "
        "FAIL (\"row count differs\"), unless it is excused as described under WARN.",
        "* **Residue.** Not treated specially: every row that the two calls return is compared, each normalised by its own peak. "
        "For EMD and CEEMDAN the reconstruction check requires the returned rows to sum to the signal, so the residue is part "
        "of the returned matrix and is compared like an IMF. For EEMD, whatever rows `eemd()` returns are compared.",
        "* **`sift a/b` (EMD only, informational; does not affect status).** Per IMF, the number of sift iterations is compared: "
        "GPU = the kernel's loop counter at exit (debug counters); PyEMD = number of `extract_max_min_spline` calls between "
        "consecutive `end_condition` calls (counted by a wrapper subclass). `b` = IMFs compared (up to the shorter of the two lists; "
        "GPU entries with a negative counter are skipped), `a` = IMFs with equal counts (an IMF that reached MAX_ITERATION "
        "counts as equal if PyEMD's count is >= MAX_ITERATION - 1, because PyEMD checks the cap before fitting). "
        "Equal counts mean both codes took the same decisions.",
        "* **PASS** -- no failure and no warning. **WARN** -- nothing failed, but the overflow flag (`EMDOverflowWarning`) was raised on "
        "a case not designed to hit the cap: for EMD, at least one IMF reached MAX_ITERATION (PyEMD's own cap; the IMF is accepted as "
        "it stands); for EEMD/CEEMDAN, some inner EMD run hit MAX_ITERATION or ran out of IMF slots. A row-count or `rel` mismatch "
        "is also downgraded to WARN when the flag was raised and the first numerically differing row is at or after the first IMF "
        "that hit MAX_ITERATION. **FAIL** -- any of: row count differs or `rel` > tolerance (not excused as above); EEMD per-order "
        "trial counts differ; the GPU library raised an exception; NaN/inf in the output; reconstruction error above its tolerance; "
        "(default stop rule, EMD) an IMF violating |#extrema - #zero-crossings| <= 1, IMFs that hit MAX_ITERATION excluded; "
        "(EMD) the overflow flag raised although no IMF hit MAX_ITERATION. **SKIP** -- PyEMD itself raised on the input "
        "(the GPU is not run unless `--probe-rejected`), or a scale case's predicted GPU time exceeded `--gpu-budget` (not run). "
        "Cases without a PyEMD reference (`gpu-only`) get only the reference-free checks. The `stats` suite is judged by its "
        "own rules (noise-moment checks, determinism, permutation energy test p-value).",
        f"* **Tolerance.** `rel` <= {args.tol:g}. The reconstruction check uses max|signal - sum(rows)| <= (1e-8 for EMD, 1e-12 for "
        "EEMD/CEEMDAN) + 1e-11 * max(signal peak, largest row peak) * number of rows.",
        f"* **Signals and seeds.** Global `--seed` = {args.seed} (shifts every signal and noise seed; default 0). Signals are deterministic "
        "functions of (kind, N, seed). Scale suites: EMD on white noise / multitone (three tones + trend) / chirp_am (per preset); "
        "EEMD and CEEMDAN on `noisy_mix` (multitone + 0.5 x white noise). Branch suites: noise, chirp_am, multitone and `pyemd_demo`. "
        "Edge suite: " + ", ".join(EDGE_SIGNALS) + ". Fuzz suite: signal and configuration are both functions of the case index "
        "(index + seed). EEMD/CEEMDAN comparisons inject identical Gaussian noise into both libraries: trial i's noise row comes from "
        f"`SeedSequence([4321 + seed, i])`, scaled as PyEMD scales its own (noise_width x peak-to-peak for EEMD, noise_scale for CEEMDAN). "
        "The `stats` suite instead uses each library's native generator (seeds 1000+i for PyEMD, 2000+i for the GPU library, on `noisy_mix`).",
        "* **CEEMDAN reference, trial order.** PyEMD's parallel CEEMDAN adds its trials with `Pool.imap_unordered` (completion "
        "order), so two parallel PyEMD runs of one input differ in the last bits, amplified to ~1e-7 with `parabol` extrema. The "
        "reference subclass uses the ordered `Pool.imap` (same parallelism and timing; trials summed in trial order = PyEMD's "
        "`parallel=False` result). PyEMD's EEMD already uses the ordered `pool.map`.",
        "* **`emd-geometry` / `api`.** `emd-geometry` launches the same decomposition with 1-8 thread blocks per signal and "
        "compares each with PyEMD (`rel` as above). `api` checks the PyEMD drop-in surface against PyEMD run in-process with "
        "`parallel=False`: PASS = bit-identical bytes, WARN = within the tolerance but not bit-identical, FAIL = beyond it, "
        "a missing/mismatched exception, or an unexpected warning.",
        "",
    ])


def build_summary(records, env, args, P, interrupted: bool) -> str:
    L: List[str] = []
    cnt = counts(records)
    overall = "FAIL" if cnt["FAIL"] else ("PASS (with warnings)" if cnt["WARN"] else "PASS")
    L.append("# GPU EMD / EEMD / CEEMDAN vs PyEMD -- validation report\n")
    L.append(f"**Overall: {overall}** -- {len(records)} cases: {cnt['PASS']} pass, {cnt['WARN']} warn, {cnt['FAIL']} fail, "
             f"{cnt['SKIP']} skipped (PyEMD itself rejects the input).  Preset `{args.preset}`, tolerance {args.tol:g} "
             f"(relative to each IMF's own peak).{'  **Run was interrupted: partial results.**' if interrupted else ''}\n")
    L.append(methodology_block(args))
    g, h = env.get("gpu", {}), env.get("host", {})
    L.append("## Environment\n")
    L.append(md_table(["item", "value"], [
        ["GPU", f"{g.get('name')} (sm_{str(g.get('compute_capability', '')).replace('.', '')}, {g.get('sm_count')} SMs, "
                f"{g.get('mem_total_gb')} GB)"],
        ["CUDA / CuPy", f"runtime {g.get('cuda_runtime')}, driver {g.get('cuda_driver')}, CuPy {g.get('cupy')}"],
        ["CPU (PyEMD)", f"{h.get('cpu')} -- {h.get('cpu_threads_usable')} usable threads (PyEMD parallel=True uses all of them)"],
        ["Software", f"PyEMD (EMD-signal) {h.get('pyemd')}, NumPy {h.get('numpy')}, SciPy {h.get('scipy')}, Python {h.get('python')}"],
    ]) + "\n")

    # ---- verdict per suite ----
    L.append("## 1. Verdict by suite\n")
    rows = []
    for s in sorted({r["suite"] for r in records}, key=_suite_key):
        rs = [r for r in records if r["suite"] == s]
        c = counts(rs)
        rows.append([s, len(rs), c["PASS"], c["WARN"], c["FAIL"], c["SKIP"], sum(r["mode"] == "gpu-only" for r in rs)])
    L.append(md_table(["suite", "cases", "pass", "warn", "FAIL", "skipped", "gpu-only (no PyEMD reference)"], rows) + "\n")

    scale = [r for r in records if r["suite"].endswith("-scale")]
    scale.sort(key=lambda r: (["emd", "eemd", "ceemdan"].index(r["algo"]), r["signal"], r["N"]))

    # ---- overflow audit ----
    L.append("## 2. Overflow audit -- is the old capacity problem gone?\n")
    L.append("`flag` = the library raised `EMDOverflowWarning` (some sift hit PyEMD's own MAX_ITERATION, or the IMF slots ran "
             "out). `cap hit` (EMD only) = at least one IMF really reached MAX_ITERATION: a property of the *signal*, "
             "shared with PyEMD. A flag **without** a cap hit would be the capacity problem and is a FAIL.\n")
    rows = [[r["algo"], r["signal"], r["N"], r["trials"] or "-", r["rows_gpu"] if r["rows_gpu"] is not None else "-",
             "**YES**" if r["flagged"] else "no", {None: "-", True: "yes", False: "no"}[r["cap_hit"]],
             r["n_overflow"] if r["n_overflow"] is not None else "-", r["status"]] for r in scale + [x for x in records if x["suite"] == "trials"]]
    L.append(md_table(["algo", "signal", "N", "trials", "GPU rows", "flag", "cap hit", "flagged inner runs", "status"], rows) + "\n")

    # ---- accuracy ----
    L.append("## 3. Accuracy vs PyEMD (default configuration, identical injected noise)\n")
    L.append("`rel` = worst over rows of max|GPU-PyEMD| / (row peak). `max abs err` = max|GPU-PyEMD| over all compared rows and "
             "samples; `rms err` = sqrt(mean((GPU-PyEMD)^2)) over the same samples (both un-normalised, report-only; "
             "see Methodology). `sift =` = IMFs whose sift-iteration count is identical "
             "(EMD): equal counts mean the GPU kernel took exactly PyEMD's decisions. `counts =` = per-order trial counts "
             "(EEMD).\n")
    rows = []
    for r in scale + [x for x in records if x["suite"] == "trials"]:
        if r["ref_status"] != "ok":
            continue
        rows.append([r["algo"], r["signal"], r["N"], r["trials"] or "-", f"{r['rows_ref']}/{r['rows_gpu']}", fmt_e(r["max_abs"]),
                     fmt_e(r["rms_err"]), fmt_e(r["rel"]), f"{r['sift_equal']}/{r['sift_cmp']}" if r["sift_cmp"] else "-",
                     {None: "-", True: "yes", False: "**NO**"}[r["counts_equal"]], r["status"]])
    L.append(md_table(["algo", "signal", "N", "trials", "rows (PyEMD/GPU)", "max abs err", "rms err", "rel", "sift =", "counts =", "status"],
                      rows) + "\n")

    # ---- gpu-only ----
    go = [r for r in scale if r["mode"] == "gpu-only"]
    if go:
        L.append("### Sizes without a PyEMD reference (too slow on CPU, or beyond --cpu-budget)\n")
        L.append("Checked against PyEMD's own definition of an IMF instead: every IMF must satisfy |#extrema - #zero-crossings| <= 1, "
                 "rows must sum back to the signal, nothing may be flagged.\n")
        rows = [[r["algo"], r["signal"], r["N"], r["trials"] or "-", r["rows_gpu"], fmt_s(r["gpu_time"]),
                 fmt_e(r["recon_err"]), r["imf_violations"] if r["imf_violations"] is not None else "-",
                 "YES" if r["flagged"] else "no", r["status"], (r["ref_note"] or "").split(" (from")[0][:80]] for r in go]
        L.append(md_table(["algo", "signal", "N", "trials", "rows", "GPU time", "reconstruction err", "IMF-property violations",
                           "flag", "status", "why no reference"], rows) + "\n")

    # ---- speed ----
    L.append("## 4. Speed\n")
    L.append("Wall clock of the PyEMD-compatible call on identical input. GPU = median of synchronised repeats after a warm-up "
             "call; PyEMD = its defaults (EEMD/CEEMDAN: `parallel=True` on all usable threads; EMD has no parallel mode).\n")
    sp = [r for r in scale + [x for x in records if x["suite"] == "trials"] if r["ref_time"] and r["gpu_time"]]
    has_serial = any(r["ref_serial_time"] for r in sp)
    rows = [[r["algo"], r["signal"], r["N"], r["trials"] or "-", fmt_s(r["ref_time"]), fmt_s(r["gpu_time"]), f"**{fmt_x(r['speedup'])}**"]
            + ([fmt_x(r["speedup_serial"])] if has_serial else []) for r in sp]
    L.append(md_table(["algo", "signal", "N", "trials", "PyEMD", "GPU", "speedup"] + (["vs 1-core PyEMD"] if has_serial else []),
                      rows) + "\n")

    # ---- branch coverage ----
    br = [r for r in records if r["suite"].endswith("-branch")]
    if br:
        L.append("## 5. Branch coverage\n")
        L.append("EMD, every compile-time kernel variant (worst `rel` over the signals/sizes run for that cell; "
                 "FAIL marks any failing case in the cell):\n")
        rows = []
        for sp_ in SPLINES:
            row = [sp_]
            for det in DETECTIONS:
                cell = [r for r in br if r["suite"] == "emd-branch" and r["config"]["emd"].get("spline_kind") == sp_
                        and r["config"]["emd"].get("extrema_detection", "simple") == det and len(r["config"]["emd"]) == 2
                        and r["config"]["T"] == "none" and r["config"]["max_imf"] == -1]
                if not cell:
                    row.append("-")
                    continue
                bad = [r for r in cell if r["status"] == "FAIL"]
                done = [r for r in cell if r["rel"] is not None]
                worst = max((r["rel"] for r in done), default=None)
                row.append(("**FAIL** " if bad else "ok ") + f"({len(cell)} run{'s' if len(cell) != 1 else ''}, rel {fmt_e(worst)})")
            rows.append(row)
        L.append(md_table(["spline_kind", "extrema_detection=simple", "extrema_detection=parabol"], rows) + "\n")
        rows = []
        for s in ("emd-branch", "eemd-branch", "ceemdan-branch"):
            rs = [r for r in br if r["suite"] == s]
            if not rs:
                continue
            c = counts(rs)
            worst = max((r["rel"] for r in rs if r["rel"] is not None), default=None)
            sif = [r for r in rs if r["sift_cmp"]]
            rows.append([s, len(rs), c["PASS"], c["WARN"], c["FAIL"], c["SKIP"], fmt_e(worst),
                         f"{sum(r['sift_equal'] for r in sif)}/{sum(r['sift_cmp'] for r in sif)}" if sif else "-",
                         sum(1 for r in rs if r["counts_varied"])])
        L.append(md_table(["suite", "cases", "pass", "warn", "FAIL", "skipped", "worst rel", "IMF sift counts equal",
                           "cases with varying per-order trial counts"], rows) + "\n")
        L.append("Everything the matrix ran:\n")
        rows = [[r["id"], r["status"], fmt_e(r["rel"]), fmt_e(r["max_abs"]), fmt_e(r["rms_err"]),
                 f"{r['rows_ref']}/{r['rows_gpu']}" if r["rows_ref"] is not None else "-",
                 "FLAG" if r["flagged"] else ""] for r in sorted(br, key=lambda r: (_suite_key(r["suite"]), r["id"]))]
        L.append("<details><summary>full branch table (" + str(len(rows)) + " cases)</summary>\n\n" +
                 md_table(["case", "status", "rel", "max abs err", "rms err", "rows", ""], rows) + "\n\n</details>\n")

    # ---- robustness ----
    rob = [r for r in records if r["suite"] in ("emd-edge", "emd-fuzz")]
    if rob:
        L.append("## 6. Robustness: edge cases and seeded random differential testing\n")
        rows = []
        for s in ("emd-edge", "emd-fuzz"):
            rs = [r for r in rob if r["suite"] == s]
            if rs:
                c = counts(rs)
                rows.append([s, len(rs), c["PASS"], c["WARN"], c["FAIL"], c["SKIP"],
                             fmt_e(max((r["rel"] for r in rs if r["rel"] is not None), default=None)),
                             sum(1 for r in rs if r["flagged"]),
                             sum(1 for r in rs if (r["amp_ratio"] or 0) > 1e6)])
        L.append(md_table(["suite", "cases", "pass", "warn", "FAIL", "skipped (PyEMD rejects)", "worst rel", "flagged",
                           "IMF > 1e6 x signal (spline blow-up; PyEMD does it too when they match)"], rows) + "\n")

    # ---- stats ----
    st = [r for r in records if r["suite"] == "stats"]
    if st:
        L.append("## 7. Native noise generators: statistical equivalence and determinism\n")
        rows = []
        for r in st:
            m = r.get("metrics", {})
            rows.append([r["id"].split("/", 1)[1], r["status"], f"{m['p']:.3f}" if "p" in m else "-",
                         f"{m['d_pyemd']:.3g} / {m['d_gpu']:.3g} / {m['d_cross']:.3g}" if "p" in m else "-",
                         "; ".join(r["reasons"])[:90]])
        L.append(md_table(["check", "status", "p-value", "mean distance PyEMD-PyEMD / GPU-GPU / cross", "note"], rows) + "\n")

    # ---- api ----
    ap = [r for r in records if r["suite"] == "api"]
    if ap:
        L.append("## 7b. PyEMD drop-in surface (api suite; PyEMD parallel=False in-process, bit-for-bit)\n")
        L.append(md_table(["check", "status", "note"],
                          [[r["id"].split("/", 1)[1], r["status"], "; ".join(r["reasons"])[:110] or "bit-identical"]
                           for r in ap]) + "\n")

    # ---- non-passing ----
    bad = [r for r in records if r["status"] in ("FAIL", "WARN")]
    L.append(f"## 8. Findings ({len(bad)} non-passing cases)\n")
    if not bad:
        L.append("None.\n")
    for r in sorted(bad, key=lambda r: (r["status"] != "FAIL", _suite_key(r["suite"]), r["id"])):
        L.append(f"* **{r['status']}** `{r['id']}` -- " + " | ".join(r["reasons"]))
        L.append(f"  * reproduce: `{repro(r, args)}`")
    L.append("\n## How to read this\n")
    L.append("* **PASS** -- same row count, every row within the tolerance, nothing flagged, reference-free checks clean.\n"
             "* **WARN** -- the output matches (or the deviation starts at an IMF that hit MAX_ITERATION) but the library raised its "
             "overflow flag; PyEMD accepts such an IMF silently.\n"
             "* **FAIL** -- a mismatch, a missing/extra IMF, an exception, a flag that is not an iteration-cap, NaN, a broken IMF.\n"
             "* **SKIP** -- PyEMD itself raised on this input (nothing to match; `--probe-rejected` runs the GPU anyway), "
             "or the GPU time was predicted to exceed `--gpu-budget` (see the case's note).\n"
             "* EEMD/CEEMDAN noise: generated once per trial from `(seed, trial)` and given to both libraries; PyEMD is run through a "
             "subclass that replaces only the noise source. Timing therefore compares identical work.\n")
    return "\n".join(L)


def make_plots(records, out_dir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    pdir = out_dir / "plots"
    pdir.mkdir(exist_ok=True)
    made: List[str] = []
    sc = [r for r in records if r["suite"].endswith("-scale") and r["gpu_time"]]
    if not sc:
        return made
    names = {"emd": "EMD (white noise)", "eemd": "EEMD", "ceemdan": "CEEMDAN"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, algo in zip(axes, ("emd", "eemd", "ceemdan")):
        rr = sorted([r for r in sc if r["algo"] == algo and (algo != "emd" or r["signal"] == "noise")], key=lambda r: r["N"])
        if not rr:
            ax.set_visible(False)
            continue
        ref = [(r["N"], r["ref_time"]) for r in rr if r["ref_time"]]
        if ref:
            ax.loglog(*zip(*ref), "o-", label="PyEMD")
        ax.loglog(*zip(*[(r["N"], r["gpu_time"]) for r in rr]), "s-", label="GPU")
        ax.set_title(names[algo] + (f", {rr[0]['trials']} trials" if rr[0]["trials"] else ""))
        ax.set_xlabel("N (samples)")
        ax.set_ylabel("seconds")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "time_vs_N.png", dpi=140)
    plt.close(fig)
    made.append("plots/time_vs_N.png")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for algo in ("emd", "eemd", "ceemdan"):
        rr = sorted([r for r in sc if r["algo"] == algo and r["speedup"] and (algo != "emd" or r["signal"] == "noise")],
                    key=lambda r: r["N"])
        if rr:
            ax.semilogx([r["N"] for r in rr], [r["speedup"] for r in rr], "o-", label=names[algo])
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("speedup  (PyEMD time / GPU time)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "speedup_vs_N.png", dpi=140)
    plt.close(fig)
    made.append("plots/speedup_vs_N.png")
    acc = [r for r in records if r["rel"] is not None and r["suite"] in ("emd-scale", "eemd-scale", "ceemdan-scale", "trials")]
    if acc:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for algo in ("emd", "eemd", "ceemdan"):
            rr = [r for r in acc if r["algo"] == algo]
            if rr:
                ax.loglog([r["N"] for r in rr], [max(r["rel"], 1e-18) for r in rr], "o", label=names[algo], alpha=0.8)
        ax.set_xlabel("N (samples)")
        ax.set_ylabel("worst relative error vs PyEMD  (0 plotted as 1e-18)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / "error_vs_N.png", dpi=140)
        plt.close(fig)
        made.append("plots/error_vs_N.png")
    return made


CSV_COLS = ["id", "suite", "algo", "N", "signal", "trials", "mode", "status", "rows_ref", "rows_gpu", "max_abs", "rel", "rel_peak",
            "sift_equal", "sift_cmp", "counts_equal", "flagged", "cap_hit", "n_overflow", "recon_err", "imf_violations",
            "ref_time", "gpu_time", "gpu_cold", "speedup", "ref_serial_time", "speedup_serial", "deterministic", "reasons"]


def write_outputs(records, env, args, P, out_dir: Path, interrupted: bool = False, final: bool = False) -> None:
    import csv
    (out_dir / "results.json").write_text(json.dumps(clean(dict(env=env, args=vars(args), preset=P, interrupted=interrupted,
                                                              records=records)), indent=1))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in records:
            w.writerow([("; ".join(r["reasons"]) if k == "reasons" else ("" if r.get(k) is None else r.get(k))) for k in CSV_COLS])
    if not final:
        return
    (out_dir / "summary.md").write_text(build_summary(records, env, args, P, interrupted), encoding="utf-8")
    with open(out_dir / "failures.txt", "w", encoding="utf-8") as f:
        if not any(r["status"] in ("FAIL", "WARN") for r in records):
            f.write("No FAIL or WARN cases.\n")
        for r in records:
            if r["status"] in ("FAIL", "WARN"):
                f.write(f"== {r['status']}  {r['id']}\n  reasons: {' | '.join(r['reasons'])}\n  reproduce: {repro(r, args)}\n"
                        f"  config: {jdump(r['config'])}\n")
                if r.get("tb"):
                    f.write("  traceback:\n" + "".join("    " + ln + "\n" for ln in r["tb"].splitlines()))
                f.write("\n")
    made = make_plots(records, out_dir)
    if made:
        log("plots: " + ", ".join(made))


# ======================================================================================================================
# CLI
# ======================================================================================================================
GROUPS = {"emd": ["emd-scale", "emd-branch", "emd-edge", "emd-fuzz", "emd-geometry"], "geometry": ["emd-geometry"],
          "api": ["api"], "eemd": ["eemd-scale", "eemd-branch"],
          "ceemdan": ["ceemdan-scale", "ceemdan-branch"], "scale": ["emd-scale", "eemd-scale", "ceemdan-scale"],
          "branch": ["emd-branch", "eemd-branch", "ceemdan-branch"], "robust": ["emd-edge", "emd-fuzz"],
          "trials": ["trials"], "stats": ["stats"]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Match + speed validation of the GPU EMD/EEMD/CEEMDAN library against PyEMD.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="standard", help="how much to run")
    ap.add_argument("--suites", default="all", help="comma list of suites or groups: " + ", ".join(sorted(set(SUITE_ORDER) | set(GROUPS))))
    ap.add_argument("--only", action="append", default=[], help="run only cases whose id contains this text (repeatable)")
    ap.add_argument("--phase", choices=["all", "cpu", "gpu"], default="all", help="cpu = PyEMD references only; gpu = GPU only")
    ap.add_argument("--tol", type=float, default=1e-8, help="PASS threshold for the per-row relative error")
    ap.add_argument("--cpu-budget", type=float, default=None, help="skip a PyEMD run predicted to exceed this many seconds (preset default)")
    ap.add_argument("--ref-max-n", type=int, default=0, help="never run PyEMD above this N (0 = no limit)")
    ap.add_argument("--gpu-budget", type=float, default=None,
                    help="skip a scale case whose GPU time is predicted (extrapolated from smaller N) to exceed this many "
                         "seconds; 0 = no limit (preset default: smoke 120, quick 600, standard 3600, full 7200)")
    ap.add_argument("--resume", default=None,
                    help="results.json of an interrupted run (same --preset/--seed): its finished cases are kept, not re-run")
    ap.add_argument("--repeats", type=int, default=None, help="max timed GPU repeats (preset default)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None, help="override the N list of the scale suites")
    ap.add_argument("--trials", type=int, default=None, help="override trials of the EEMD/CEEMDAN scale suites")
    ap.add_argument("--trials-sweep", type=int, nargs="+", default=None, help="override the trials list of the 'trials' suite")
    ap.add_argument("--fuzz", type=int, default=None, help="override the number of random differential cases")
    ap.add_argument("--serial-baseline", action="store_true", help="also time PyEMD EEMD/CEEMDAN with parallel=False")
    ap.add_argument("--serial-max-n", type=int, default=16000, help="largest N for --serial-baseline")
    ap.add_argument("--probe-rejected", action="store_true", help="also run the GPU on inputs PyEMD itself rejects")
    ap.add_argument("--seed", type=int, default=0, help="shifts every signal/noise seed (fresh data, fresh cache keys)")
    ap.add_argument("--out", default=str(HERE / "results"), help="output root")
    ap.add_argument("--cache", default=str(HERE / "ref_cache"), help="PyEMD reference cache")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write the reference cache")
    ap.add_argument("--lib-dir", default=None, help="directory with emd.py/eemd.py/ceemdan.py/kernels.py (default: next to this file)")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the quick GPU smoke test that precedes the CPU phase")
    ap.add_argument("--force", action="store_true", help="continue even if the smoke test fails")
    ap.add_argument("--list", action="store_true", help="print the case list and exit")
    ap.add_argument("--gpu-smoke", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def select_plan(plan: List[Case], args) -> Tuple[List[Case], bool]:
    want = set()
    for tok in [t.strip() for t in args.suites.split(",") if t.strip()]:
        if tok == "all":
            want |= set(SUITE_ORDER)
        elif tok in GROUPS:
            want |= set(GROUPS[tok])
        elif tok in SUITE_ORDER:
            want.add(tok)
        else:
            raise SystemExit(f"unknown suite {tok!r}; choose from {sorted(set(SUITE_ORDER) | set(GROUPS))}")
    plan = [c for c in plan if c.suite in want]
    if args.only:
        plan = [c for c in plan if any(o in c.id for o in args.only)]
    want_stats = "stats" in want and (not args.only or any(o in "stats" for o in args.only))
    args.want_api = "api" in want and (not args.only or any(o in "api" for o in args.only))
    return plan, want_stats


def main(argv=None) -> int:
    global SEED
    args = parse_args(argv)
    SEED = int(args.seed)
    if _PYEMD_ERR is not None:
        print(f"PyEMD could not be imported ({_PYEMD_ERR!r}). Install the reference library:  pip install EMD-signal", file=sys.stderr)
        return 2
    P = dict(PRESETS[args.preset])
    if args.sizes:
        P["emd_sizes"] = P["ens_sizes"] = sorted(set(args.sizes))
        P["emd_gpu_only"], P["ens_gpu_only"] = [], []
    if args.trials:
        P["scale_trials"] = args.trials
    if args.trials_sweep:
        P["trials_sweep"] = sorted(set(args.trials_sweep))
    if args.fuzz is not None:
        P["fuzz"] = args.fuzz
    if args.repeats:
        P["repeats"] = args.repeats
    args.cpu_budget = args.cpu_budget or P["cpu_budget"]
    if args.gpu_budget is None:
        args.gpu_budget = {"smoke": 120.0, "quick": 600.0, "standard": 3600.0, "full": 7200.0}[args.preset]
    elif args.gpu_budget <= 0:
        args.gpu_budget = float("inf")
    plan, args.want_stats = select_plan(build_plan(P), args)
    if "stats" in [t.strip() for t in args.suites.split(",")] and P["stats_runs"] < 3:
        P["stats_runs"] = 6                       # asked for by name: honour it even in the 'smoke' preset
    args.want_stats = args.want_stats and P["stats_runs"] >= 3

    if args.gpu_smoke:
        return gpu_smoke(args)
    if args.list:
        for c in ordered(plan, GPU_RANK):
            print(f"{c.suite:15s} {c.ref_policy:7s} tier{c.tier}  {c.id}")
        by = {}
        for c in plan:
            by[c.suite] = by.get(c.suite, 0) + 1
        print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(by.items(), key=lambda kv: _suite_key(kv[0]))) +
              f"  (+ statistical suite: {'yes' if args.want_stats else 'no'})  total {len(plan)} cases")
        return 0

    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.open(out_dir / "run.log")
    log(f"output: {out_dir}\ncommand: {' '.join(sys.argv)}")
    do_cpu, do_gpu = args.phase in ("all", "cpu"), args.phase in ("all", "gpu")
    cache = RefCache(Path(args.cache), enabled=not args.no_cache)
    env: Dict[str, Any] = dict(host=host_info(), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"host: {env['host']['cpu']} ({env['host']['cpu_threads_usable']} threads), PyEMD {env['host']['pyemd']}, "
        f"{len(plan)} cases selected from preset '{args.preset}'")

    if do_gpu and not args.skip_smoke:               # fail fast, in a child process so CUDA stays out of this one
        log("\n--- GPU smoke test (child process) ---")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--gpu-smoke", "--tol", str(args.tol)]
        if args.lib_dir:
            cmd += ["--lib-dir", args.lib_dir]
        rc = subprocess.run(cmd).returncode
        if rc != 0 and not args.force:
            log("GPU smoke test FAILED -- fix that first (or --force / --skip-smoke). Nothing else was run.")
            return 2
    skipped: Dict[str, str] = {}
    records: List[Dict[str, Any]] = []
    if args.resume:
        try:
            records.extend(json.loads(Path(args.resume).read_text(encoding="utf-8")).get("records", []))
            log(f"resume: loaded {len(records)} finished cases from {args.resume}")
        except Exception as e:
            log(f"--resume: could not read {args.resume}: {e}")
            return 2
    interrupted = False
    try:
        if do_cpu:
            selfcheck_injection()
            skipped = phase_reference(plan, args, cache, P)
        if do_gpu:
            load_gpu_libs(args.lib_dir)
            env["gpu"] = gpu_info()
            log(f"\nGPU: {env['gpu']}")
            finish = lambda recs, partial=False: write_outputs(recs, env, args, P, out_dir, final=False)
            phase_gpu(plan, args, P, cache, skipped, finish, records)
    except KeyboardInterrupt:
        interrupted = True
        log("\n*** interrupted: writing partial results ***")
    except FatalGPUError as e:
        interrupted = True
        log(f"\n*** {e}\n*** remaining GPU cases skipped; earlier results are valid and are written below ***")
    except AssertionError as e:
        log(f"\nHARNESS SELF-CHECK FAILED: {e}")
        return 2
    finally:
        if do_gpu and records:
            write_outputs(records, env, args, P, out_dir, interrupted, final=True)
    if not do_gpu:
        log("\nCPU phase finished. Run again without --phase cpu (or with --phase gpu) on the GPU machine.")
        return 2 if interrupted else 0
    cnt = counts(records)
    word = "STOPPED EARLY" if interrupted else "DONE"
    log(f"\n=== {word}: {len(records)} cases -- {cnt['PASS']} pass, {cnt['WARN']} warn, {cnt['FAIL']} FAIL, {cnt['SKIP']} skipped ===")
    if records:
        log(f"report: {out_dir / 'summary.md'}")
    if interrupted:
        return 1 if cnt["FAIL"] else 2
    return 1 if cnt["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

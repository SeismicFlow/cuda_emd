# cuda_emd

**EMD, EEMD and CEEMDAN on NVIDIA GPUs, bit-for-bit identical to PyEMD and with the PyEMD interface.**

`cuda_emd` is a CUDA implementation of empirical mode decomposition (EMD) and of its noise-assisted variants
EEMD and CEEMDAN for Python. It reproduces the output of the `EMD`, `EEMD` and `CEEMDAN` classes of
[PyEMD](https://github.com/laszukdawid/PyEMD) 1.10.0 (PyPI package `EMD-signal`) in every bit, for every
configuration of the reference, and it exposes the same classes, constructor arguments, methods and output
format. An existing PyEMD script changes only its import line:

```python
# from PyEMD import EMD, EEMD, CEEMDAN
from cuda_emd import EMD, EEMD, CEEMDAN
```

- **Identical results.** Every floating-point operation and every decision of the reference (extrema and
  zero-crossing counts, stopping tests, number of IMFs) is reproduced in IEEE-754 double precision. The IMFs,
  their number and the residue are the ones PyEMD computes, not approximations of them.
- **Complete method family.** EMD, EEMD and CEEMDAN, all seven envelope types (`cubic`, `akima`, `pchip`,
  `cubic_hermite`, `quadratic`, `linear`, `slinear`), both extrema-detection modes (`simple`, `parabol`),
  all stopping rules (thresholds, `FIXE`, `FIXE_H`, `MAX_ITERATION`) and arbitrary time axes.
- **Batches.** Many signals, or all trials of an ensemble, are decomposed in a single kernel launch; long
  signals are distributed over several streaming multiprocessors.
- **Plain installation.** Pure Python plus CUDA C++ source that is compiled at run time through CuPy and
  NVRTC. No compiler toolchain, no build step, the same files on Windows and Linux.

The method, its validation and its performance are described in the accompanying article (see
[Citation](#citation)).

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Interface](#interface)
- [How exactness is obtained](#how-exactness-is-obtained)
- [Validation](#validation)
- [Performance](#performance)
- [Reproducing the results of the article](#reproducing-the-results-of-the-article)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA GPU with CUDA support and double-precision arithmetic; kernel compilation verified for compute capabilities 6.1 (Pascal), 8.6 (Ampere) and 12.0 (Blackwell) |
| Driver | NVIDIA driver supporting the CUDA version of the installed CuPy wheel |
| Python | 3.10 or later |
| Required packages | NumPy; CuPy (binary wheel for your CUDA major version, e.g. `cupy-cuda12x`) |
| Optional packages | SciPy (probes the rounding of the installed reference stack); PyEMD / `EMD-signal` 1.10.0 (comparison scripts, `Visualisation`, CPU-side helper methods) |

The results in the article were obtained with Python 3.12.10, NumPy 2.4.6, SciPy 1.18.0, CuPy 14.1.1 (CUDA
runtime and NVRTC 12.9) and PyEMD 1.10.0 on Windows 10 with a GeForce GTX 1080 Ti.

## Installation

```bash
# 1. CuPy for your CUDA version (only the NVIDIA driver must be present on the system)
pip install cupy-cuda12x          # CUDA 12.x; use cupy-cuda13x for CUDA 13.x

# 2. cuda_emd
pip install git+https://github.com/SeismicFlow/cuda_emd
```

Optional extras install CuPy and the reference stack in the same step:

```bash
pip install "cuda_emd[cuda12,reference] @ git+https://github.com/SeismicFlow/cuda_emd"
```

If no CUDA toolkit is installed, CuPy can provide the run-time compiler NVRTC as a wheel
(`pip install "cupy-cuda12x[ctk]"`; see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)).

Check the installation on your machine:

```python
from cuda_emd import EMD
print(EMD().selftest())   # compares the envelopes of the compiled kernel with SciPy's on this machine
```

The kernel is compiled on its first use for each configuration, which takes a few seconds; CuPy caches the
compiled module on disk for later processes. Call `warmup()` before timing.

## Quick start

### Drop-in use (NumPy in, NumPy out)

```python
import numpy as np
from cuda_emd import EMD, EEMD, CEEMDAN

t = np.linspace(0, 1, 4096)
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 40 * t) + 0.1 * np.random.randn(t.size)

imfs = EMD()(x)                        # rows: IMFs, then the residue (as in PyEMD)

eemd = EEMD(trials=100)
eemd.noise_seed(12345)                 # PyEMD's numpy noise stream: same seed, same result as PyEMD
eimfs = eemd(x)

cimfs = CEEMDAN(trials=100, epsilon=0.005)(x)

emd = EMD(spline_kind="akima", extrema_detection="parabol", nbsym=3, std_thr=0.1)
imfs = emd(x, t)                       # any configuration of the reference
```

### Batch use (data stays on the GPU)

```python
import cupy as cp
from cuda_emd import GPUBatchEMD, GPUBatchEEMD, GPUBatchCEEMDAN

signals = cp.random.standard_normal((512, 4096))         # B signals of length N, float64
imfs, n_imfs, errors = GPUBatchEMD()(signals)            # imfs: (B, rows, N); signal i: imfs[i, :n_imfs[i]]

mean_imfs, counts, n_errors = GPUBatchEEMD(trials=100)(signals[0])   # noise drawn on the GPU
cimfs, n_errors = GPUBatchCEEMDAN(trials=100)(signals[0])
```

`errors` is non-zero only for a signal on which PyEMD itself raises an exception; `EMD.emd()` raises the same
exception type as PyEMD.

## Interface

| Class | Input / output | Purpose |
|---|---|---|
| `EMD`, `EEMD`, `CEEMDAN` | NumPy arrays | Drop-in replacements for the PyEMD classes |
| `GPUBatchEMD`, `GPUBatchEEMD`, `GPUBatchCEEMDAN` | CuPy arrays | Batch interface for pipelines that stay on the GPU |

The drop-in classes accept every constructor argument of the corresponding PyEMD class, with PyEMD's defaults:

- **EMD:** `spline_kind`, `nbsym`, `extrema_detection`, `std_thr`, `svar_thr`, `energy_ratio_thr`,
  `total_power_thr`, `range_thr`, `FIXE`, `FIXE_H`, `MAX_ITERATION`
- **EEMD:** `trials`, `noise_width`, `ext_EMD`, `separate_trends`, `noise_kind` and all EMD arguments
- **CEEMDAN:** `trials`, `epsilon`, `ext_EMD`, `noise_scale`, `noise_kind`, `beta_progress`, `range_thr`,
  `total_power_thr` and all EMD arguments

and provide PyEMD's methods and attributes: `emd()`, `eemd()`, `ceemdan()`, `get_imfs_and_residue()`,
`get_imfs_and_trend()`, `noise_seed()`, `ensemble_count()`, `ensemble_std()`, `all_imfs`, `imfs`, `residue`.
A two-dimensional input to `EMD()(signals)` returns the decomposition of every row. PyEMD's plotting helper is
available as `cuda_emd.Visualisation` when PyEMD is installed.

GPU-specific options:

| Option | Classes | Meaning |
|---|---|---|
| `blocks_per_signal` | all | thread blocks per signal; `None` (default) selects it automatically |
| `noise_backend` | EEMD, CEEMDAN | `"numpy"`: PyEMD's noise stream (default of the drop-in classes); `"cupy"`: noise drawn on the GPU (default of the batch classes) |
| `chunk_trials` | EEMD, CEEMDAN | trials per launch; default: as many as fit in device memory |
| `store_all_imfs` | EEMD | keep the IMFs of every trial (needed for `all_imfs`, `ensemble_std()`) |
| `warmup()` | all | compile the kernel before the first timed call |
| `selftest()` | EMD | check the envelopes against SciPy on the current machine |

With the same noise, EEMD and CEEMDAN return the result of PyEMD in its serial mode, in which the trials are
accumulated in trial order. A noise array can also be passed explicitly: `eemd(S, noise=...)`,
`ceemdan(S, noise=...)`.

## How exactness is obtained

The decomposition is defined by a procedure whose discrete decisions (which samples are extrema, when a sift
stops, how many IMFs there are) turn last-bit differences into different IMFs. `cuda_emd` therefore replays the
arithmetic of the reference instead of approximating it:

- **Binary64 arithmetic in the reference's order.** Only correctly rounded basic operations are used,
  multiply-add contraction is disabled (`--fmad=false`), and no mathematical library function is called. The
  result does not depend on the GPU model or on the launch configuration.
- **Exact replay of NumPy's pairwise summation**, which decides the stopping tests, with the same blocking and
  recursion tree, evaluated in parallel.
- **Verified chunked replay of LAPACK `dgtsv`** for the cubic-spline envelopes: the sequential elimination
  with partial pivoting is split into chunks that start from a warmed-up state, every chunk boundary is
  certified bit for bit against the exact state, and chunks that disagree are re-solved. The result equals the
  sequential solve.
- **Integer evaluation of sign and order predicates** on the IEEE-754 bit patterns where this provably gives
  the result of the floating-point expression.
- **Probing of the installed reference stack.** Whether OpenBLAS fuses multiply-adds in the solves of the
  quadratic and three-knot envelopes depends on the host CPU, and SciPy's Akima formula depends on its version;
  both are probed once per process and the matching kernel variant is compiled.

Details are given in the article.

## Validation

The implementation was compared with PyEMD 1.10.0 on a GeForce GTX 1080 Ti:

| Test set | Content | Result |
|---|---|---|
| Differential test campaign | 432 cases: signal lengths from 2 to 2²¹ samples, all envelope types and extrema modes, all stopping rules, ensemble options, degenerate and extreme signals (amplitudes 10⁻²⁰⁰ to 10²⁰⁰), 100 random configurations, 1 to 8 blocks per signal, interface checks | all 418 cases with a PyEMD reference bitwise identical (2,979 output rows, 1.15 × 10⁸ samples); identical number of sift iterations for every IMF of the EMD suites |
| Real data | MIT-BIH Arrhythmia Database, record 208 (lead MLII): EMD, EEMD and CEEMDAN (100 trials) on 2,048-sample windows | all 58 decompositions bitwise identical |

Relative error, absolute error and RMSE are exactly zero in every compared case.

## Performance

Median wall-clock times on a GeForce GTX 1080 Ti and an Intel Core i7-7700K (GPU times end to end, including
host–device transfers; outputs bitwise identical to PyEMD in every case).

**EMD of a single signal**

| N | PyEMD | cuda_emd | Speed-up |
|---|---|---|---|
| 1,000 | 6.4–23.0 ms | 1.9–5.2 ms | 3.2–4.4 |
| 16,000 | 78.9 ms–1.19 s | 6.2–106 ms | 11–13 |
| 131,072 | 0.92–15.9 s | 21.4–301 ms | 43–58 |
| 1,048,576 | 7.9 s–11.1 min | 0.14–10.4 s | 57–74 |

Ranges over three signal families (white noise, multitone, amplitude-modulated chirp).

**EMD of batches** (N = 1,024): 512 signals take 45.2 ms, 315 times faster than PyEMD on one core and 16.6
times faster than the fastest CPU baseline using eight cores.

**EEMD and CEEMDAN with 100 trials** (PyEMD with 8 worker processes)

| Method | N | PyEMD | cuda_emd | Speed-up |
|---|---|---|---|---|
| EEMD | 1,000 | 3.18 s | 12.2 ms | 260 |
| EEMD | 131,072 | 18.1 min | 41.5 s | 26 |
| CEEMDAN | 1,000 | 9.58 s | 33.7 ms | 284 |
| CEEMDAN | 65,536 | 12.5 min | 20.1 s | 37 |

The article also compares `cuda_emd` with the Python libraries `emd`, PySDKit and PyHHT; it was faster than each
of them in every configuration benchmarked.

## Reproducing the results of the article

The folder `paper/` contains the scripts and result files of the article:

| Script | Purpose |
|---|---|
| `test_gpu_vs_pyemd.py` | differential validation against PyEMD (Section 5) |
| `real_data_validation.py` | ECG study (Section 5.4) |
| `bench.py` | run time, scaling, memory and comparison with other libraries (Section 6) |
| `pyemd_profile.py` | profile of the PyEMD reference (Section 2.3) |

The scripts import the library modules directly, so run them with the package folder on the module path:

```bash
pip install scipy EMD-signal==1.10.0 emd pysdkit pyhht      # reference and baseline libraries

# Linux / macOS shell
PYTHONPATH=cuda_emd python paper/test_gpu_vs_pyemd.py --lib-dir cuda_emd
PYTHONPATH=cuda_emd python paper/real_data_validation.py
PYTHONPATH=cuda_emd python paper/bench.py

# Windows (cmd)
set PYTHONPATH=cuda_emd
python paper\test_gpu_vs_pyemd.py --lib-dir cuda_emd
python paper\real_data_validation.py
python paper\bench.py
```

`python paper/test_gpu_vs_pyemd.py --help` lists the suites and options of the validation harness;
`bench.py --quick` runs a short version of the benchmark. The torchHHT baseline of `bench.py` requires a clone of
its repository (see the variable `HHT_DIR` in the script).

## Repository layout

```
cuda_emd/
├── cuda_emd/            the package
│   ├── __init__.py
│   ├── emd.py           EMD, GPUBatchEMD
│   ├── eemd.py          EEMD, GPUBatchEEMD
│   ├── ceemdan.py       CEEMDAN, GPUBatchCEEMDAN
│   └── kernels.py       CUDA kernel source, kernel variants and launch planning
├── paper/               scripts and result files of the article
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Citation

If you use `cuda_emd`, please cite the article and the software:

> M. Mahzad, Bit-Exact GPU Acceleration of Empirical Mode Decomposition, EEMD and CEEMDAN in Python: A Batched
> CUDA Implementation of PyEMD (submitted).

```bibtex
@software{mahzad_cuda_emd,
  author  = {Mahzad, Matin},
  title   = {cuda\_emd: {EMD}, {EEMD} and {CEEMDAN} on {NVIDIA} {GPUs}, bit-for-bit identical to {PyEMD}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/SeismicFlow/cuda_emd},
  license = {MIT}
}
```

Please also cite PyEMD, whose results `cuda_emd` reproduces:

> D. Laszuk, Python implementation of empirical mode decomposition algorithm, 2017.
> https://github.com/laszukdawid/PyEMD, doi:10.5281/zenodo.5459184

## License

MIT License, see [LICENSE](LICENSE).

Contact: Matin Mahzad, matinmahzad@yahoo.com, ORCID [0009-0000-9346-8451](https://orcid.org/0009-0000-9346-8451)

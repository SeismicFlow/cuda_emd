"""bench.py -- timing benchmark for Reviewer 2: batch sizes x signal lengths x repeats x GPU memory x baselines (PyEMD, PySDKit, emd/Quinn, torchHHT, pyhht).

Put it next to emd.py / eemd.py / ceemdan.py / kernels.py. Run  `python\\python.exe bench.py --quick`  first, then without --quick.
Writes bench_results.csv (a row per measurement, saved as it goes) and bench_env.txt (versions/hardware for the paper).
Timing: GPU rows = one untimed warm-up, then REPS timed runs, GPU synced before and after each.
  CPU baselines = no warm-up, CPU_REPS timed runs (pool workers are started and run once beforehand).
  Ensembles = ours: one untimed call, baselines: none; ENS_REPS timed runs, no explicit sync.
  (dev) = data already on the GPU, kernel only        (e2e) = numpy in -> numpy out, includes host<->device copies
Rows marked "est": a CPU baseline timed on CPU_SUBSET signals and scaled linearly to B (serial work is linear in B).
"""
import sys, os, time, csv, platform, subprocess, warnings
import importlib.metadata as md
import multiprocessing as mp
import numpy as np

QUICK = "--quick" in sys.argv
HHT_DIR = "Hilbert-Huang-transform"   # EDIT if needed: the cloned torchHHT repo (the folder that contains torchHHT/)
NS, BS = ([1024, 4096], [1, 16]) if QUICK else ([1024, 4096, 16384], [1, 16, 128, 512])
REPS, CPU_REPS, CPU_SUBSET = (3 if QUICK else 5), 3, 8
ENS_NS, ENS_REPS, TRIALS = ([1024], 2, 50) if QUICK else ([1024, 4096], 2, 100)
NOISE = 0.1   # std of the noise added to every test signal (Reviewer 1 asked for noisy input)


# ---- top-level so Windows worker processes can import them (heavy imports stay inside main) ----
_E = None
def _init():
    global _E
    from PyEMD import EMD
    _E = EMD()
def _one(x):
    return _E.emd(x)

def load_quinn():
    """Quinn's PyPI package `emd` has the same import name as our local emd.py (GPUBatchEMD), so a plain `import emd`
    returns the local file. Load the installed package from site-packages under the alias `quinn_emd` instead."""
    import importlib.util, logging
    pkg = str(md.distribution("emd").locate_file("emd"))
    spec = importlib.util.spec_from_file_location("quinn_emd", os.path.join(pkg, "__init__.py"), submodule_search_locations=[pkg])
    mod = importlib.util.module_from_spec(spec); sys.modules["quinn_emd"] = mod; spec.loader.exec_module(mod)
    for n in ("emd", "quinn_emd"): logging.getLogger(n).setLevel(logging.ERROR)   # it logs every sift call at INFO
    return mod
_Q = None
def _init_q():
    global _Q
    import warnings; warnings.simplefilter("ignore")   # workers don't inherit main()'s filter; emd emits numpy warnings
    _Q = load_quinn().sift.sift
def _one_q(x):
    return _Q(x)


def _cold_child():
    """Run in a FRESH process via `bench.py --cold-child` (item 3). Times the first-ever GPU EMD at B=1, N=1024:
    GPUBatchEMD() construction + H2D + first call + D2H. Python imports are outside the timer; g.warmup() is NOT called."""
    warnings.simplefilter("ignore")
    import cupy as cp
    from emd import GPUBatchEMD
    X = make(1, 1024)
    t0 = time.perf_counter()
    g = GPUBatchEMD()
    o = g(cp.asarray(X)); cp.cuda.Device().synchronize()
    o[0].get(); o[1].get()
    print("COLD_S %.6f" % (time.perf_counter() - t0))


def make(B, N, seed=0):
    r = np.random.default_rng(seed); t = np.linspace(0, 1, N)
    f, ph, a = r.uniform([3, 15, 60], [8, 35, 150], (B, 3)), r.uniform(0, 6.28, (B, 3)), (1, .6, .3)
    X = sum(a[k] * np.sin(2 * np.pi * f[:, k:k + 1] * t + ph[:, k:k + 1]) for k in range(3))
    return np.ascontiguousarray(X + NOISE * r.standard_normal((B, N)))


def clock(fn, reps, sync=lambda: None):
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return ts


def main():
    import cupy as cp
    warnings.simplefilter("ignore")
    from PyEMD import EMD, EEMD, CEEMDAN
    from emd import GPUBatchEMD
    from eemd import GPUBatchEEMD
    from ceemdan import GPUBatchCEEMDAN
    try:
        import torch; sys.path.insert(0, HHT_DIR); from torchHHT.hht import emd as thht_emd
    except Exception as e:
        torch = thht_emd = None; print("torchHHT unavailable, skipping it:", repr(e))
    try:
        import pysdkit as sdk
        sd = {n: getattr(sdk, n, None) for n in ("EMD", "EEMD", "CEEMDAN")}
    except Exception as e:
        sd = {}; print("pysdkit unavailable, skipping it:", repr(e))
    try:
        import scipy
        if not hasattr(scipy, "angle"): scipy.angle = np.angle   # pyhht 0.1.0 does `from scipy import angle`, which modern scipy removed
        from pyhht.emd import EMD as HHTEMD   # PyPI `pyhht` (EMD only); imported as pyhht.emd so it doesn't clash with local emd.py
    except Exception as e:
        HHTEMD = None; print("pyhht unavailable, skipping it:", repr(e))
    try:
        qs = load_quinn().sift   # emd (Quinn), loaded under an alias because local emd.py shadows the name
    except Exception as e:
        qs = None; print("emd (Quinn) unavailable, skipping it:", repr(e))

    # ---- environment (Reviewer 2, point 1) ----
    def ver(p):
        try: return md.version(p)
        except Exception: return "n/a"
    try: commit = subprocess.check_output(["git", "-C", HHT_DIR, "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "n/a"
    prop = cp.cuda.runtime.getDeviceProperties(0)
    env = [f"GPU: {prop['name'].decode()}  CUDA driver/runtime: {cp.cuda.runtime.driverGetVersion()}/{cp.cuda.runtime.runtimeGetVersion()}",
           f"CPU: {platform.processor()}  logical cores: {os.cpu_count()}  OS: {platform.platform()}",
           f"Python {platform.python_version()}  numpy {np.__version__}  cupy {cp.__version__}  torch {torch.__version__ if torch else 'n/a'}",
           f"EMD-signal {ver('EMD-signal')}  emd(Quinn) {ver('emd')}  pysdkit {ver('pysdkit')}  pyhht {ver('pyhht')}  torchHHT commit {commit}",
           f"signals: 3 random sines + N(0,{NOISE}) noise, float64; reps GPU={REPS} CPU={CPU_REPS} ensembles={ENS_REPS}; trials={TRIALS}",
           f"pool workers: {os.cpu_count()} (os.cpu_count(), logical cores; used by PyEMD pool{' and emd(Quinn) pool' if qs else ''}; chunksize=1)"]
    print("\n".join(env)); open("bench_env.txt", "w").write("\n".join(env) + "\n")
    print("\n".join([
        "methodology:",
        f"  warm-up: GPU (ours, torchHHT) = 1 untimed run + sync per (B,N), then {REPS} timed dev runs, then {REPS} timed e2e runs (no separate e2e warm-up);"
        " ours also gets g.warmup() once at start. CPU baselines = no warm-up call (pool workers are started and run once on 256-sample signals beforehand)."
        f" Ensembles: ours 1 untimed call, baselines none; {ENS_REPS} timed runs.",
        "  sync: GPU timed runs call device synchronize (cupy Device.synchronize / torch.cuda.synchronize) before starting and before stopping the timer."
        " CPU runs and ensemble runs pass no sync (ensemble: the call is timed as it returns).",
        f"  '+-' = population std (numpy, ddof=0) of the individual timed runs: {REPS} for GPU rows, {CPU_REPS} for CPU rows, {ENS_REPS} for ensembles;"
        " the main number is the median of the same runs. For est rows the std is of the scaled times.",
        "  dev = compute only, data already on the GPU (no host->device copy, no device->host copy)."
        " e2e = numpy in -> host->device copy + compute + device->host copy -> numpy out.",
        f"  est = CPU baseline timed on the first k=min(B,cap) signals ({CPU_REPS} runs; cap={CPU_SUBSET} for the 1-core baselines, PySDKit and pyhht,"
        f" cap={4 * os.cpu_count()} = 4 x workers for the pools), each run's time multiplied by B/k (assumes cost linear in B)."
        " Rows with k==B are measured directly and carry no tag. The k-signal timing is cached and reused for every B with the same (baseline, N, k)."]))

    out = open("bench_results.csv", "w", newline=""); w = csv.writer(out)
    w.writerow(["lib", "algo", "B", "N", "median_s", "std_s", "reps", "est", "gpu_mem_MB"])
    rows = []
    def rec(lib, algo, B, N, ts, est="", mem=""):
        m, s = float(np.median(ts)), float(np.std(ts))
        rows.append((lib, algo, B, N, m)); w.writerow([lib, algo, B, N, f"{m:.6f}", f"{s:.6f}", len(ts), est, mem]); out.flush()
        print(f"{algo:8}{lib:18} B={B:<4} N={N:<6} {m * 1e3:10.1f} ms +-{s * 1e3:7.1f}  {est} {mem + ' MB' if mem else ''}")

    # ---- backends ----
    pool = cp.get_default_memory_pool(); E = EMD()
    g = GPUBatchEMD(); g.warmup()
    ours = dict(prep=lambda X: cp.asarray(X), run=lambda d: g(d), fetch=lambda o: (o[0].get(), o[1].get()),
                sync=cp.cuda.Device().synchronize, reset=pool.free_all_blocks, mem=lambda: pool.total_bytes() / 2**20)
    tor = torch and dict(prep=lambda X: torch.as_tensor(X, device="cuda"), run=lambda d: thht_emd(d, num_imf=10),
                         fetch=lambda o: o.cpu().numpy(), sync=torch.cuda.synchronize,
                         reset=lambda: (torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()),
                         mem=lambda: torch.cuda.max_memory_allocated() / 2**20)
    W = os.cpu_count(); P = mp.Pool(W, initializer=_init); P.map(_one, list(make(W, 256)))  # start workers once
    P2 = None
    cpu = {"PyEMD 1core": (lambda X: [E.emd(x) for x in X], CPU_SUBSET),
           "PyEMD pool": (lambda X: P.map(_one, list(X), chunksize=1), 4 * W)}
    if sd.get("EMD"): cpu["PySDKit"] = (lambda X: [sd["EMD"]().fit_transform(x) for x in X], CPU_SUBSET)
    if HHTEMD: cpu["pyhht"] = (lambda X: [HHTEMD(x).decompose() for x in X], CPU_SUBSET)   # serial only, like PySDKit
    if qs:
        P2 = mp.Pool(W, initializer=_init_q); P2.map(_one_q, list(make(W, 256)))   # each worker loads Quinn's emd itself
        cpu["emd(Quinn) 1core"] = (lambda X: [qs.sift(x) for x in X], CPU_SUBSET)
        cpu["emd(Quinn) pool"] = (lambda X: P2.map(_one_q, list(X), chunksize=1), 4 * W)

    x = make(1, 1024)[0]; ref, got = E.emd(x), g.emd(x)   # sanity: are we timing the same decomposition as PyEMD?
    print("sanity: PyEMD rows", ref.shape[0], "| ours rows", got.shape[0], "| max|diff|",
          np.abs(ref - got).max() if ref.shape == got.shape else "shape differs")
    if HHTEMD: print("sanity: pyhht rows", HHTEMD(x).decompose().shape[0], "(its output is IMFs x samples, residual included)")
    if qs: print("sanity: emd(Quinn) rows", qs.sift(x).shape[1], "(its output is samples x IMFs)")
    try:   # batched check: ASSUMES g(d) returns (imfs[B,K,N], n_imfs[B]) -- the same two arrays fetch() reads above; verify against emd.py
        X16 = make(16, 1024); o = g(cp.asarray(X16)); A, K = o[0].get(), o[1].get()
        refs = [E.emd(xi) for xi in X16]; ae = re = 0.0; bad = 0
        for i in range(16):
            gi = A[i, :int(K[i])]
            if gi.shape != refs[i].shape: bad += 1; continue   # different IMF count: excluded from the error stats
            d = np.abs(gi - refs[i]).max(); ae = max(ae, d); re = max(re, d / np.abs(refs[i]).max())
        print(f"sanity (batched B=16, N=1024 vs 16 independent PyEMD runs): max abs err {ae:.3e} | max rel err {re:.3e}"
              f" (per-signal max|diff| / max|PyEMD output|) | rows with different IMF count: {bad}/16 (excluded) | ours output shapes {A.shape}, {K.shape}")
    except Exception as e:
        print("sanity (batched) could not run:", repr(e), "-- check the layout of g(d)'s outputs in emd.py")
    o = None   # drop the GPU arrays from the batched check so they don't inflate the gpu_mem_MB column

    def gpu_bench(name, be, X):
        B, N = X.shape; d = None
        try:
            be["reset"](); d = be["prep"](X); be["run"](d); be["sync"]()          # warm-up
            ts_dev = clock(lambda: be["run"](d), REPS, be["sync"])
            mem = "%.0f" % be["mem"]()
            ts_e2e = clock(lambda: be["fetch"](be["run"](be["prep"](X))), REPS, be["sync"])
            rec(name + " (dev)", "EMD", B, N, ts_dev, mem=mem); rec(name + " (e2e)", "EMD", B, N, ts_e2e, mem=mem)
        except Exception as e:
            print(f"  SKIP {name} B={B} N={N}: {type(e).__name__}: {str(e)[:90]}")
        finally:
            d = None; pool.free_all_blocks()
            if torch: torch.cuda.empty_cache()

    # ---- EMD: batch sizes x signal lengths ----
    cache = {}
    for N in NS:
        for B in BS:
            X = make(B, N)
            for name, (fn, cap) in cpu.items():
                k = min(B, cap)
                if (name, N, k) not in cache:
                    try: cache[(name, N, k)] = clock(lambda: fn(X[:k]), CPU_REPS)
                    except Exception as e: cache[(name, N, k)] = None; print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:90]}")
                if cache[(name, N, k)]:
                    rec(name, "EMD", B, N, [t * B / k for t in cache[(name, N, k)]], est="est" if k < B else "")
            gpu_bench("ours", ours, X)
            if tor: gpu_bench("torchHHT", tor, X)

    # ---- cold start: one fresh process, B=1, N=1024, printed next to the warmed steady-state e2e time ----
    res = None
    try:
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--cold-child"], capture_output=True, text=True, timeout=900,
                             env={**os.environ, "CUPY_CACHE_IN_MEMORY": "1"})   # in-memory cupy kernel cache => kernels are really compiled
        cold = float([l for l in res.stdout.splitlines() if l.startswith("COLD_S")][-1].split()[1])
        warm = next((r[4] for r in rows if r[:4] == ("ours (e2e)", "EMD", 1, 1024)), None)
        print(f"cold start (fresh process, B=1 N=1024, GPUBatchEMD() + first call incl. CUDA context, kernel compile, H2D/D2H; no disk kernel cache): {cold * 1e3:.0f} ms"
              + (f"  |  steady-state (same case, e2e, median of {REPS} runs after 1 untimed warm-up): {warm * 1e3:.1f} ms" if warm else ""))
    except Exception as e:
        print("cold-start measurement failed:", repr(e), (res.stderr[-300:] if res else ""))

    # ---- EEMD / CEEMDAN: only PyEMD and PySDKit also have them (torchHHT and pyhht are EMD-only) ----
    print("EEMD/CEEMDAN benchmarked at B=1 (single-signal API)")
    def ens(lib, algo, N, fn, warm=False, gpu=False):
        try:
            if gpu: pool.free_all_blocks()
            if warm: fn()
            ts = clock(fn, ENS_REPS)
            rec(lib, algo, 1, N, ts, mem="%.0f" % (pool.total_bytes() / 2**20) if gpu else "")
        except Exception as e:
            print(f"  SKIP {lib} {algo} N={N}: {type(e).__name__}: {str(e)[:90]}")
    for N in ENS_NS:
        x = make(1, N)[0]
        for algo, G, gm, PE, pm, qm in (("EEMD", GPUBatchEEMD, "eemd", EEMD, "eemd", "ensemble_sift"),
                                        ("CEEMDAN", GPUBatchCEEMDAN, "ceemdan", CEEMDAN, "ceemdan", "complete_ensemble_sift")):
            o = getattr(G(trials=TRIALS), gm)
            ens("ours (e2e)", algo, N, lambda: o(x), warm=True, gpu=True)
            for par in ([True, False] if N == ENS_NS[0] else [True]):   # serial PyEMD only at the smallest N (slow)
                p = getattr(PE(trials=TRIALS, parallel=par), pm)
                ens("PyEMD " + ("parallel" if par else "serial"), algo, N, lambda: p(x))
            if sd.get(algo):
                try: s = sd[algo](trials=TRIALS)
                except TypeError: s = sd[algo]()
                ens("PySDKit", algo, N, lambda: s.fit_transform(x))
            if qs:   # serial only: its nprocesses>1 workers can't import the aliased module (see notes)
                q = getattr(qs, qm)
                ens("emd(Quinn) serial", algo, N, lambda: q(x, nensembles=TRIALS, nprocesses=1))

    print("\nratio = baseline time / ours (e2e); >1 means ours is faster")
    for key in sorted({r[1:4] for r in rows}):
        t = {r[0]: r[4] for r in rows if r[1:4] == key}
        if "ours (e2e)" in t:
            print(key, "  ".join(f"{k}: {v / t['ours (e2e)']:.1f}x" for k, v in t.items() if not k.startswith("ours")))
    out.close(); P.close()
    if P2: P2.close()


if __name__ == "__main__":
    _cold_child() if "--cold-child" in sys.argv else main()

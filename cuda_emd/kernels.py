"""
kernels.py -- the CUDA kernel of the GPU batch EMD library, and the host-side launcher around it.

DESIGN (v3): EXACT PyEMD ARITHMETIC AND EXACT PyEMD DECISIONS, ON ANY NUMBER OF SMs PER SIGNAL
---------------------------------------------------------------------------------------------
The kernel reproduces PyEMD 1.x (EMD.emd) step by step, BIT FOR BIT: compiled with fused multiply-adds off, every
floating-point operation rounds exactly as numpy / scipy round it -- the extrema, knots, spline coefficients,
envelopes, the sifted signal, the residue -- AND every quantity a decision is taken on is the very number numpy
computes: the sums of check_imf and end_condition are numpy's pairwise sums (same 128-element blocks, same 8
accumulators, same recursion tree), max(imf_old) / min(imf_old) follow Python's builtin semantics, np.max / np.min
propagate NaN.  The discrete decisions of EMD (extrema counts, zero crossings, when a sift stops, how many IMFs there
are) are therefore PyEMD's by construction, not "with overwhelming probability", and they do not depend on the launch
geometry (threads per block, blocks per signal) either.

What runs where, per sift iteration (a signal is processed by a GROUP of G thread blocks, G chosen per call):

  1. extrema        every thread scans a contiguous chunk of the signal (PyEMD's _find_extrema_simple rules,
                    plateau quirks included; or _find_extrema_parabol); a group-wide scan turns per-chunk counts
                    into global ranks; a second walk writes the extrema.  Sign / order / equality tests on finite,
                    not-tiny samples are decided on the IEEE bit patterns with integer instructions (provably the same
                    result as the floating-point expression; full-rate on GeForce GPUs, whose FP64 rate is 1/32-1/64).
  2. mirroring      PyEMD's prepare_points_simple / prepare_points_parabol: one thread, O(nbsym) work.
  3. spline         the envelope through the knots, in whichever spline_kind was asked for:
                    * cubic: scipy's CubicSpline(not-a-knot) = LAPACK dgtsv, a pivoting first-order recurrence that
                      cyclic reduction cannot reproduce (it re-associates the arithmetic).  It is solved as a
                      SPECULATIVE CHUNKED RECURRENCE WITH AN EXACTNESS CERTIFICATE: each thread eliminates a chunk of
                      rows after warming its entry state up from earlier rows; every chunk boundary is then verified
                      bit for bit; chunks whose entry state disagrees are re-solved from the exact state.  The
                      result is provably identical to the serial solve.  Back substitution works the same way.
                      Three knots: PyEMD's own cubic_spline_3pts (numpy's 3x3 dgesv, replicated).
                    * pchip / akima / cubic_hermite: derivatives independent per knot (fully parallel).
                    * linear / slinear: closed forms per piece.
                    * quadratic: scipy's make_interp_spline(k=2) = LAPACK dgbsv, replayed by one thread per envelope.
  4. sift update    new = old - 0.5*(max_env + min_env); integer positions are differenced as integers (exact).
  5. stop test      extrema / zero-crossing counts of the new signal (also the next iteration's input); check_imf's
                    statistics are formed only when PyEMD looks at them (|extNo - nzm| < 2 and the envelope signs ok).
  6. IMF done       output row, running sum (np.sum(IMF, axis=0) order), residue, end_condition.

LAUNCH GEOMETRY (adapts to the GPU; the results do not depend on it)
    The launcher reads the GPU (SM count, resident blocks per SM for this kernel) and picks, per call:
    * threads per block: ~N/16, more when the batch is too small to fill the GPU;
    * blocks per signal G: 1 while the batch fills the GPU; when it does not and the signals are long, each signal
      gets G > 1 blocks on different SMs, synchronised by a software group barrier (all blocks of a launch are
      co-resident by construction: batch * G never exceeds what the GPU holds at once).  A single long signal thus
      uses the whole GPU -- 28 SMs on a GTX 1080 Ti, 170 on an RTX 5090 -- instead of one SM.

EXACTNESS AND ITS LIMITS
    The reference is PyEMD 1.10 running on the INSTALLED numpy / scipy.  What the bit-for-bit claim relies on:
    * IEEE-754 double arithmetic with fused multiply-add disabled (--fmad=false); explicit fma only where the installed
      BLAS uses it: whether OpenBLAS fuses multiply-adds in numpy's 3x3 solve (3-knot cubic envelopes) and in scipy's
      banded LU (quadratic spline) depends on the CPU kernel it selects (e.g. Haswell kernels: no; SkylakeX: yes).
      The launcher PROBES the installed numpy and scipy once (_p3_fma_form / _quad_fma_form) and compiles to match.
    * akima: scipy changed the breakpoint-slope expression between 1.16 and 1.17; probed likewise (_akima_form).
    * scipy's CubicSpline squares the end intervals with pow(x, 2.0); the kernel computes x*x.  Identical for the
      integer knot positions of "simple" extrema (the default); for "parabol" positions a C library whose pow(x, 2) is
      not correctly rounded could differ in the last bit of the two end pieces.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import cupy as cp

# PyEMD's spline_kind / extrema_detection options and the integers the CUDA source dispatches on.
SPLINE_KINDS = {"cubic": 0, "linear": 1, "slinear": 2, "quadratic": 3, "cubic_hermite": 4, "pchip": 5, "akima": 6}
EXTREMA_MODES = {"simple": 0, "parabol": 1}

# per-signal error bits of the batch output `errors` (0 = a normal decomposition)
ERR_NOEXT = 1      # PyEMD raises IndexError: more than two extrema, but all maxima or all minima
ERR_SPLINE = 2     # PyEMD / scipy raise while building an envelope (too few / non-increasing knots, singular system)
ERR_CAP = 4        # internal capacity (cannot happen)
ERR_NONTERM = 8    # the decomposition did not end within N + 2 IMFs (PyEMD would not terminate either)

_INT32_MAX = 2**31 - 1
_VS_DOUBLES = 8  # per-thread doubles of solver verification state (must equal VS_DOUBLES in the CUDA source)


def _i32(value) -> "np.int32":
    """A Python int as the int32 the kernel takes, clamped to the int32 range (PyEMD accepts any int)."""
    return np.int32(max(-_INT32_MAX - 1, min(_INT32_MAX, int(value))))


def _align4(x: int) -> int:
    return (int(x) + 3) & ~3


# ----------------------------------------------------------------------------------------------------------------
# Which arithmetic does the INSTALLED numpy / scipy use?  (probed once per process, bit for bit)
# ----------------------------------------------------------------------------------------------------------------
def _fma(a: float, b: float, c: float) -> float:
    """Correctly rounded a*b + c (math.fma on Python >= 3.13; exact rational arithmetic otherwise)."""
    f = getattr(math, "fma", None)
    if f is not None:
        return f(a, b, c)
    r = Fraction(a) * Fraction(b) + Fraction(c)
    return float(r) if r != 0 else (a * b + c)


def _mac(use_fma: bool):
    return _fma if use_fma else (lambda a, b, c: a * b + c)


_AKIMA_FORM_CACHE: list = []
_P3_CACHE: list = []
_QUAD_CACHE: list = []


def _akima_form() -> int:
    """Which expression does the INSTALLED scipy use for the breakpoint slopes of Akima1DInterpolator?
        0:  (f1*m[i+1] + f2*m[i+2]) / f12                     scipy 1.13 .. 1.16
        1:  m[i+1] + (f2/f12) * (m[i+2] - m[i+1])             scipy >= 1.17"""
    if _AKIMA_FORM_CACHE:
        return _AKIMA_FORM_CACHE[0]
    form = 1
    try:
        from scipy.interpolate import Akima1DInterpolator

        rng = np.random.RandomState(2024)
        x = np.arange(48, dtype=np.float64)
        y = rng.standard_normal(48)
        s_ref = np.asarray(Akima1DInterpolator(x, y).c[2])           # derivative at the left knot of every piece
        m = np.empty(x.size + 3)
        m[2:-2] = np.diff(y) / np.diff(x)
        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        m[-2] = 2.0 * m[-3] - m[-4]
        m[-1] = 2.0 * m[-2] - m[-3]
        dm = np.abs(np.diff(m))
        f1, f2 = dm[2:], dm[:-2]
        f12 = f1 + f2
        i = np.arange(x.size)
        old = ((f1 * m[i + 1] + f2 * m[i + 2]) / f12)[: s_ref.size]
        new = (m[i + 1] + (f2 / f12) * (m[i + 2] - m[i + 1]))[: s_ref.size]
        if np.array_equal(new, s_ref):
            form = 1
        elif np.array_equal(old, s_ref):
            form = 0
        else:
            warnings.warn("The installed scipy's Akima1DInterpolator matches neither known derivative formula "
                          "bit for bit; spline_kind='akima' results may differ from it in the last bit.",
                          RuntimeWarning, stacklevel=3)
    except Exception:  # scipy not installed / unexpected internals: take the current formula
        form = 1
    _AKIMA_FORM_CACHE.append(form)
    return form


def _p3_solve(x, y, use_fma):
    """The kernel's three_point_setup (numpy's dgesv on PyEMD's 3x3 system), in Python."""
    f = _mac(use_fma)
    x0, x1, x2 = x
    y0, y1, y2 = y
    x1x0, x2x1, y1y0, y2y1 = x1 - x0, x2 - x1, y1 - y0, y2 - y1
    ia, ib = 1.0 / x1x0, 1.0 / x2x1
    v1 = ((3.0 * y1y0) * ia) * ia
    v3 = ((3.0 * y2y1) * ib) * ib
    v2 = v1 + v3
    a00, a01, a02 = 2.0 * ia, ia, 0.0
    a10, a11, a12 = ia, 2.0 * (ia + ib), ib
    a20, a21, a22 = 0.0, ib, 2.0 * ib
    r0 = 1.0 / a00
    a10 = a10 * r0
    a20 = a20 * r0
    a11 = a11 - f(a10, a01, 0.0)
    a21 = a21 - f(a20, a01, 0.0)
    r1 = 1.0 / a11
    a21 = a21 * r1
    a12 = a12 - f(a10, a02, 0.0)
    s = f(a20, a02, 0.0)
    s = f(a21, a12, s)
    a22 = a22 - s
    b0, b1, b2 = v1, v2, v3
    b1 = f(-b0, a10, b1)
    b2 = f(-b0, a20, b2)
    b2 = f(-b1, a21, b2)
    b2 = b2 / a22
    b0 = f(-b2, a02, b0)
    b1 = f(-b2, a12, b1)
    b1 = b1 / a11
    b0 = f(-b1, a01, b0)
    b0 = b0 / a00
    return (b0, b1, b2)


def _p3_fma_form() -> int:
    """Does the INSTALLED numpy's 3x3 solve (np.linalg.solve -> OpenBLAS dgesv) fuse its multiply-adds?  Decided by
    comparing np.linalg.solve with both replays on a fixed set of PyEMD three-point systems, bit for bit."""
    if _P3_CACHE:
        return _P3_CACHE[0]
    rng = np.random.RandomState(31337)
    hits = [0, 0]
    total = 0
    try:
        for i in range(64):
            if i % 2:
                x = np.sort(rng.rand(3) * 50.0)
            else:
                x = np.cumsum(rng.randint(1, 30, 3)).astype(np.float64)
            y = rng.randn(3) * np.exp(rng.randn() * 3.0)
            if not (x[1] > x[0] and x[2] > x[1]):
                continue
            x1x0, x2x1 = x[1] - x[0], x[2] - x[1]
            _a, _b = 1.0 / x1x0, 1.0 / x2x1
            M = np.array([[2 * _a, _a, 0], [_a, 2.0 * (_a + _b), _b], [0, _b, 2.0 * _b]])
            v1 = 3 * (y[1] - y[0]) * _a * _a
            v3 = 3 * (y[2] - y[1]) * _b * _b
            k = tuple(np.linalg.solve(M, np.array([v1, v1 + v3, v3]).T).tolist())
            total += 1
            for m in (0, 1):
                hits[m] += int(_p3_solve(tuple(x.tolist()), tuple(y.tolist()), bool(m)) == k)
    except Exception:
        total = 0
    if total and hits[1] == total:
        form = 1
    elif total and hits[0] == total:
        form = 0
    else:
        form = 1
        if total:
            warnings.warn("numpy's 3x3 solve matches neither the fused nor the unfused LAPACK replay bit for bit; "
                          "3-knot cubic envelopes may differ from PyEMD in the last bit.", RuntimeWarning, stacklevel=3)
    _P3_CACHE.append(form)
    return form


def _quad_replay(x, y, use_fma):
    """The kernel's quad_setup + quad_factor_solve (scipy make_interp_spline(k=2) -> dgbsv), in Python."""
    f = _mac(use_fma)
    n = len(x)

    def qt(i):
        if i <= 2:
            return x[0]
        if i >= n:
            return x[n - 1]
        return (x[i - 1] + x[i - 2]) / 2.0

    def find_piece(v):
        lo, hi = 0, n - 3
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if qt(mid + 2) <= v:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def basis(tm1, t0, t1, t2, v):
        if t1 == t0:
            a0 = a1 = 0.0
        else:
            w = 1.0 / (t1 - t0)
            a0 = 0.0 + w * (t1 - v)
            a1 = w * (v - t0)
        h0 = 0.0
        if t1 == tm1:
            h1 = 0.0
        else:
            w = a0 / (t1 - tm1)
            h0 = h0 + w * (t1 - v)
            h1 = w * (v - tm1)
        if t2 == t0:
            h2 = 0.0
        else:
            w = a1 / (t2 - t0)
            h1 = h1 + w * (t2 - v)
            h2 = w * (v - t0)
        return h0, h1, h2

    rows = []
    for i in range(n):
        pc = find_piece(x[i])
        l = pc + 2
        rows.append((basis(qt(l - 1), qt(l), qt(l + 1), qt(l + 2), x[i]), pc))

    def row(i, jref):
        (h0, h1, h2), pc = rows[i]
        o = pc - jref
        r = [0.0] * 5
        r[o], r[o + 1], r[o + 2] = h0, h1, h2
        return r

    b = [float(v) for v in y]
    A, P, Q = row(0, 0), row(1, 0), row(2, 0)
    U = []
    ju = 0
    for j in range(n):
        km = min(n - 1 - j, 2)
        jp, pm = 0, abs(A[0])
        if km >= 1 and abs(P[0]) > pm:
            jp, pm = 1, abs(P[0])
        if km >= 2 and abs(Q[0]) > pm:
            jp, pm = 2, abs(Q[0])
        ju = max(ju, min(j + 2 + jp, n - 1))
        if jp == 1:
            A, P = P, A
            b[j], b[j + 1] = b[j + 1], b[j]
        elif jp == 2:
            A, Q = Q, A
            b[j], b[j + 2] = b[j + 2], b[j]
        if km > 0:
            uo = ju - j
            r = 1.0 / A[0]
            P[0] = P[0] * r
            if km == 2:
                Q[0] = Q[0] * r
            for c in range(1, 5):
                if uo >= c:
                    y0 = -1.0 * A[c]
                    P[c] = f(y0, P[0], P[c])
                    if km == 2:
                        Q[c] = f(y0, Q[0], Q[c])
            yb = -1.0 * b[j]
            b[j + 1] = f(yb, P[0], b[j + 1])
            if km == 2:
                b[j + 2] = f(yb, Q[0], b[j + 2])
        U.append(A[:])
        A = P[1:] + [0.0]
        P = Q[1:] + [0.0]
        Q = row(j + 3, j + 1) if j + 3 < n else [0.0] * 5
    for jj in range(n - 1, -1, -1):
        v = b[jj] / U[jj][0]
        b[jj] = v
        for t in range(1, min(jj, 4) + 1):
            q = jj - t
            b[q] = f(-v, U[q][t], b[q])
    return b


def _quad_fma_form() -> int:
    """Does the INSTALLED scipy's banded LU (make_interp_spline(k=2) -> OpenBLAS dger / daxpy) fuse multiply-adds?"""
    if _QUAD_CACHE:
        return _QUAD_CACHE[0]
    hits = [0, 0]
    total = 0
    try:
        from scipy.interpolate import make_interp_spline

        rng = np.random.RandomState(4242)
        for i in range(24):
            n = int(rng.randint(5, 60))
            if i % 2 == 0:
                x = np.cumsum(rng.randint(1, 9, size=n)).astype(np.float64)
            else:
                x = np.sort(rng.rand(n) * 100.0)
            if np.unique(x).size < n:
                continue
            y = rng.randn(n) * np.exp(rng.randn())
            ref = np.asarray(make_interp_spline(x, y, k=2).c).tolist()
            total += 1
            xl, yl = x.tolist(), y.tolist()
            for m in (0, 1):
                hits[m] += int(_quad_replay(xl, yl, bool(m)) == ref)
    except Exception:
        total = 0
    if total and hits[1] == total:
        form = 1
    elif total and hits[0] == total:
        form = 0
    else:
        form = 1
        if total:
            warnings.warn("scipy's make_interp_spline(k=2) matches neither the fused nor the unfused LAPACK replay bit "
                          "for bit; spline_kind='quadratic' results may differ from PyEMD in the last bit.",
                          RuntimeWarning, stacklevel=3)
    _QUAD_CACHE.append(form)
    return form


# ----------------------------------------------------------------------------------------------------------------
# numpy's pairwise-summation tree for an N-element float64 sum (np.add.reduce of a contiguous 1-D array)
# ----------------------------------------------------------------------------------------------------------------
_PW_CACHE: dict = {}


def pairwise_schedule(N: int) -> np.ndarray:
    """int32 array: [L, nlvl, root, leaf starts (L+1), level offsets (nlvl+1), ops (dst, left, right) * (L-1)].
    Leaves are numpy's base blocks (n <= 128: 8 interleaved accumulators, or plain order when n < 8), in position
    order; internal node k gets id L + k; ops are grouped by height so that one level is independent work."""
    N = int(N)
    hit = _PW_CACHE.get(N)
    if hit is not None:
        return hit
    leaves = []
    ops = []   # (left node, right node, height); internal node k gets id L + k

    def build(off, n):
        if n <= 128:
            leaves.append(off)
            return ("L", len(leaves) - 1, 0)
        n2 = n // 2
        n2 -= n2 % 8
        lnode = build(off, n2)
        rnode = build(off + n2, n - n2)
        h = max(lnode[2], rnode[2]) + 1
        ops.append((lnode, rnode, h))
        return ("I", len(ops) - 1, h)

    root = build(0, N)
    L = len(leaves)

    def nid(node):
        return node[1] if node[0] == "L" else L + node[1]

    nlvl = max((o[2] for o in ops), default=0)
    by_level = [[] for _ in range(nlvl)]
    for k, (l, r, h) in enumerate(ops):
        by_level[h - 1].append((L + k, nid(l), nid(r)))
    lvoff = [0]
    flat = []
    for lv in by_level:
        flat.extend(lv)
        lvoff.append(len(flat))
    out = [L, nlvl, nid(root)] + leaves + [N] + lvoff + [v for t in flat for v in t]
    arr = np.asarray(out, dtype=np.int32)
    _PW_CACHE[N] = arr
    return arr


def numpy_pairwise_sum(a) -> float:
    """Reference implementation (for tests): numpy's float64 pairwise sum, in Python."""
    a = [float(v) for v in a]

    def pw(lo, n):
        if n < 8:
            r = 0.0
            for i in range(lo, lo + n):
                r = r + a[i]
            return r
        if n <= 128:
            r = a[lo:lo + 8]
            i = 8
            while i < n - (n % 8):
                for j in range(8):
                    r[j] = r[j] + a[lo + i + j]
                i += 8
            res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n:
                res = res + a[lo + i]
                i += 1
            return res
        n2 = n // 2
        n2 -= n2 % 8
        return pw(lo, n2) + pw(lo + n2, n - n2)

    return pw(0, len(a))


_CUDA_SOURCE = r'''// ============================================================================================================
//  GPU batched EMD.  Reproduces PyEMD 1.10 (EMD.emd) arithmetic step by step -- every value AND every decision.
//  Compile-time constants substituted by the host:
//      @THREADS@ @NBSYM@ @SPLINE_KIND@ @EXTREMA_MODE@ @AKIMA_FORM@ @QUAD_FMA@ @P3_FMA@
//  Build with fused multiply-add DISABLED (--fmad=false): every floating-point operation must round exactly as the
//  numpy / scipy reference does (explicit fused multiply-adds appear only where the installed BLAS uses them).
//
//  A signal is processed by a GROUP of G thread blocks (G = 1: one block per signal, block-local barriers; G > 1: a
//  long signal spread over several SMs, with a software group barrier).  "Group thread" GTID / GSIZE replace
//  threadIdx.x / blockDim.x everywhere work is partitioned; SYNC() is the group barrier.
// ============================================================================================================
#define THREADS       @THREADS@      // compiled maximum block size (multiple of 32); a launch may use fewer threads
#define NBSYM         @NBSYM@        // PyEMD nbsym
#define SPLINE_KIND   @SPLINE_KIND@  // 0 cubic, 1 linear, 2 slinear, 3 quadratic, 4 cubic_hermite, 5 pchip, 6 akima
#define EXTREMA_MODE  @EXTREMA_MODE@ // 0 simple, 1 parabol
#define AKIMA_FORM    @AKIMA_FORM@   // akima breakpoint slope: 0 = (f1*m1 + f2*m2)/f12 (scipy <= 1.16), 1 = m1 + (f2/f12)*(m2 - m1)
#define QUAD_FMA      @QUAD_FMA@     // 1: the installed scipy's BLAS (dger/daxpy) fuses multiply-adds in the banded LU (quadratic spline)
#define P3_FMA        @P3_FMA@       // 1: the installed numpy's BLAS fuses multiply-adds in the 3x3 solve (3-knot cubic envelopes)
#define NW_MAX        (THREADS / 32)
#define FRONT         (NBSYM + 2)    // extrema (per list end) the mirroring logic may look at
#define R0            FRONT          // knot slot of the first real extremum (mirrored points live to its left)
#define VS_DOUBLES    8              // per-thread doubles of solver verification state kept in global scratch
#if SPLINE_KIND == 3
#define REC_W         5              // per-knot doubles of the solver records (quadratic: one banded-LU row of U)
#else
#define REC_W         4              // per-knot doubles of the solver records (cubic: D, U, L, B)
#endif
#define ENV_KC        (3 + REC_W)    // per envelope and knot slot: kx, ky, sv and REC_W record doubles
#if EXTREMA_MODE == 0 && SPLINE_KIND != 3
#define USE_START_TABLE 1            // integer knot positions: the piece of every chunk's first sample comes from a table
#define INTPOS        1              // ... and positions / piece boundaries are compared and differenced as integers (exact)
#else
#define USE_START_TABLE 0            // arbitrary positions / midpoint pieces: binary search
#define INTPOS        0
#endif
#define SAFE_EXP      560            // biased exponent from which sign/order predicates are evaluated on the bit patterns
#define LL_MIN        (-0x7fffffffffffffffLL - 1)
#define LL_MAX        0x7fffffffffffffffLL
#define ERR_NOEXT     1              // PyEMD raises IndexError (more than two extrema, all of one kind)
#define ERR_SPLINE    2              // PyEMD / scipy raise while building an envelope (too few / non-increasing knots, ...)
#define ERR_CAP       4              // internal capacity (cannot happen: the knot arrays hold every possible extremum)

typedef double real_t;
#define ALIGN4(x)     (((x) + 3) & ~3)
#if EXTREMA_MODE == 0
#define XPOS(TT, j)   ((real_t)(j))     // simple mode: positions are the sample indices
#else
#define XPOS(TT, j)   ((TT)[j])         // parabol mode: positions come from the (normalised) time array
#endif
#if SPLINE_KIND == 1 || SPLINE_KIND == 2 || SPLINE_KIND == 3
#define KNOTS_STRICT  0                 // interp1d kinds: scipy sorts the knots instead of rejecting unsorted ones
#else
#define KNOTS_STRICT  1                 // CubicSpline / PCHIP / Akima / CubicHermite raise unless x is strictly increasing
#endif
#if EXTREMA_MODE == 0
#define MIRROR_POINTS(S, TT, N, nmax, nmin, a, b, c, d)   mirror_simple(S, N, nmax, nmin, a, b, c, d)
#else
#define MIRROR_POINTS(S, TT, N, nmax, nmin, a, b, c, d)   mirror_points(S, TT, N, nmax, nmin, a, b, c, d)
#endif
#if P3_FMA
#define P3F(a, b, c)  __fma_rn((a), (b), (c))
#else
#define P3F(a, b, c)  ((a) * (b) + (c))      // two roundings (fmad is off)
#endif
#if QUAD_FMA
#define QF(a, b, c)   __fma_rn((a), (b), (c))
#else
#define QF(a, b, c)   ((a) * (b) + (c))
#endif

// @HELPERS_BEGIN
__device__ __forceinline__ long long dbits(real_t x) { return __double_as_longlong(x); }
__device__ __forceinline__ bool bz(long long b) { return (b & LL_MAX) == 0; }                     // x == 0.0 (either sign)
__device__ __forceinline__ bool bsafe(long long b) {                                               // finite, |x| >= 2^(SAFE_EXP-1023)
    const int e = (int)(((unsigned long long)b >> 52) & 0x7ffULL); return e >= SAFE_EXP && e != 0x7ff;
}
__device__ __forceinline__ bool bnan(long long b) { return (b & LL_MAX) > 0x7ff0000000000000LL; }
// order key: for non-NaN x, key(x) < key(y) <=> x < y (both zeros map to 0)
// (written as a select: NVVM turns  b ^ ((b >> 63) & LL_MAX)  into a bfe.u64 that keeps only the sign bit -- wrong keys
//  for every negative number on the GPU, while every CPU compiler gets it right)
__device__ __forceinline__ long long okey(long long b) { if (bz(b)) return 0; return (b < 0) ? (b ^ LL_MAX) : b; }
__device__ __forceinline__ real_t unkey(long long k) { return __longlong_as_double(k < 0 ? (k ^ LL_MAX) : k); }

// @HELPERS_END

// ---- group-wide broadcast state (shared memory when G == 1, global memory when G > 1) --------------------------
struct Ctl {
    int bad, degenerate, unsorted, err, nanf, pad0;
    int kb[2], kn[2];                   // per envelope: first knot slot and knot count
    int flag[4];                        // (unused, kept for layout stability)
    int rep[2];                         // solver fix-up rounds: "some chunk was re-solved" (round parity)
    int fidx[2][FRONT], bidx[2][FRONT]; // first / last FRONT extrema of each list (index; 0 = maxima, 1 = minima)
    real_t fval[2][FRONT], bval[2][FRONT];
    real_t fpos[2][FRONT], bpos[2][FRONT];   // parabol mode: positions
    real_t stat[4];
};

// ---- block-local shared state ------------------------------------------------------------------------------------
struct BSh {
    int wtot[3][NW_MAX];                // warp totals of the 3-way scan
    long long kred[4][NW_MAX];          // warp partials of the generic reduction
    int ired[NW_MAX];
    Ctl ctl;                            // the group state when G == 1
    int start_iv[2][THREADS];           // piece containing the first sample of each thread's chunk (G == 1)
    Ctl* c;                             // -> ctl or the group's global copy
    int* siv; int sivs;                 // start table base and per-envelope stride
    long long* gx;                      // group exchange slots: [2 parities][G][8 words]
    real_t* pwP;                        // pairwise-sum lane partials: [2 parities][L * 8 * 3]
    real_t* pwN;                        // pairwise-sum tree nodes of THIS block: [2L * 3]
    unsigned* bar;                      // group barrier words: count, generation
    int G, gb, gs, gt0, par, warm_f, warm_b, minchunk;
};

#ifdef CPU_SHIM
#define BSH (*reinterpret_cast<BSh*>(tl_block->shared))
#else
__shared__ BSh bsh_;
#define BSH bsh_
#endif
#define CTL   (BSH.c)
#define GSIZE (BSH.gs)
#define GTID  (BSH.gt0 + (int)threadIdx.x)
#define BLOCK_BARRIER() __syncthreads()

__device__ __forceinline__ unsigned ld_gen(const unsigned* p) {
#ifdef CPU_SHIM
    std::this_thread::yield();
    return ld_volatile(p);
#else
    return *(const volatile unsigned*)p;
#endif
}

// Group barrier: a block barrier when G == 1; otherwise the scheme of cooperative groups' grid sync: ONE atomic per
// block and a poll.  Block 0 of the group adds 0x80000000 - (G - 1), every other block adds 1, so the G arrivals add
// exactly 0x80000000: bit 31 of the counter flips once per barrier and the low bits return to their value (no reset,
// no wrap-around).  A block waits until bit 31 differs from the value its own atomic returned.  Critical path: one
// fence + one atomic round trip + one poll (the earlier generation-word barrier needed ~5 serialized L2 round trips).
__device__ __forceinline__ void SYNC()
{
    __syncthreads();
    if (BSH.G > 1) {
        if (threadIdx.x == 0) {
            unsigned* cnt = BSH.bar;
            const unsigned add = (BSH.gb == 0) ? (0x80000000u - (unsigned)(BSH.G - 1)) : 1u;
            __threadfence();                                        // release this block's writes
            const unsigned old = atomicAdd(cnt, add);
            while (((old ^ ld_gen(cnt)) & 0x80000000u) == 0u) { }
            __threadfence();                                        // acquire the other blocks' writes
        }
        __syncthreads();
    }
}

// ---- group-wide 3-way exclusive scan of non-negative ints ----------------------------------------------------
__device__ void group_scan3(int a, int b, int c, int* ea, int* eb, int* ec, int* ta, int* tb, int* tc) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int ia = a, ib = b, ic = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int xa = __shfl_up_sync(0xffffffffu, ia, o);
        int xb = __shfl_up_sync(0xffffffffu, ib, o);
        int xc = __shfl_up_sync(0xffffffffu, ic, o);
        if (lane >= o) { ia += xa; ib += xb; ic += xc; }
    }
    if (lane == 31) { BSH.wtot[0][w] = ia; BSH.wtot[1][w] = ib; BSH.wtot[2][w] = ic; }
    BLOCK_BARRIER();
    int pa = 0, pb = 0, pc = 0, sa = 0, sb = 0, sc = 0;
    for (int k = 0; k < nw; ++k) {
        int x0 = BSH.wtot[0][k], x1 = BSH.wtot[1][k], x2 = BSH.wtot[2][k];
        if (k < w) { pa += x0; pb += x1; pc += x2; }
        sa += x0; sb += x1; sc += x2;
    }
    if (BSH.G > 1) {                                    // block totals -> group prefix and totals
        const int p = BSH.par;
        long long* gx = BSH.gx + (size_t)p * BSH.G * 8;
        if (threadIdx.x == 0) { gx[BSH.gb * 8 + 0] = sa; gx[BSH.gb * 8 + 1] = sb; gx[BSH.gb * 8 + 2] = sc; }
        SYNC();
        int oa = 0, ob = 0, oc = 0, A = 0, Bt = 0, C = 0;
        for (int k = 0; k < BSH.G; ++k) {
            const int x0 = (int)gx[k * 8 + 0], x1 = (int)gx[k * 8 + 1], x2 = (int)gx[k * 8 + 2];
            if (k < BSH.gb) { oa += x0; ob += x1; oc += x2; }
            A += x0; Bt += x1; C += x2;
        }
        pa += oa; pb += ob; pc += oc; sa = A; sb = Bt; sc = C;
        if (threadIdx.x == 0) BSH.par = p ^ 1;
    }
    *ea = pa + ia - a; *eb = pb + ib - b; *ec = pc + ic - c;
    *ta = sa; *tb = sb; *tc = sc;
    BLOCK_BARRIER();   // wtot / par reusable
}

// ---- group-wide reduction: two int64 sums, an int64 max and min, an OR of flags.  Order-independent by design ----
struct GRed { long long s, s2, mx, mn; int f; };
__device__ __forceinline__ GRed gred_id() { GRed r; r.s = 0; r.s2 = 0; r.mx = LL_MIN; r.mn = LL_MAX; r.f = 0; return r; }
__device__ __forceinline__ void gred_add(GRed& r, long long s, long long s2, long long mx, long long mn, int f) {
    r.s += s; r.s2 += s2; r.mx = (mx > r.mx) ? mx : r.mx; r.mn = (mn < r.mn) ? mn : r.mn; r.f |= f;
}
__device__ GRed group_reduce(GRed r)
{
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const long long s = __shfl_xor_sync(0xffffffffu, r.s, o);
        const long long s2 = __shfl_xor_sync(0xffffffffu, r.s2, o);
        const long long mx = __shfl_xor_sync(0xffffffffu, r.mx, o);
        const long long mn = __shfl_xor_sync(0xffffffffu, r.mn, o);
        const int f = __shfl_xor_sync(0xffffffffu, r.f, o);
        gred_add(r, s, s2, mx, mn, f);
    }
    if (lane == 0) { BSH.kred[0][w] = r.s; BSH.kred[1][w] = r.s2; BSH.kred[2][w] = r.mx; BSH.kred[3][w] = r.mn; BSH.ired[w] = r.f; }
    BLOCK_BARRIER();
    GRed t = gred_id();
    for (int k = 0; k < nw; ++k) gred_add(t, BSH.kred[0][k], BSH.kred[1][k], BSH.kred[2][k], BSH.kred[3][k], BSH.ired[k]);
    if (BSH.G > 1) {
        const int p = BSH.par;
        long long* gx = BSH.gx + (size_t)p * BSH.G * 8;
        if (threadIdx.x == 0) {
            long long* q = gx + BSH.gb * 8;
            q[0] = t.s; q[1] = t.s2; q[2] = t.mx; q[3] = t.mn; q[4] = t.f;
        }
        SYNC();
        t = gred_id();
        for (int k = 0; k < BSH.G; ++k) { const long long* q = gx + k * 8; gred_add(t, q[0], q[1], q[2], q[3], (int)q[4]); }
        if (threadIdx.x == 0) BSH.par = p ^ 1;
    }
    BLOCK_BARRIER();
    return t;
}

// max / min of a[0..N) with an order-independent reduction on order keys; nan = 1 if any element is NaN.
__device__ void group_maxmin(const real_t* __restrict__ a, int N, real_t& mx, real_t& mn, int& nan)
{
    GRed r = gred_id();
    for (int j = GTID; j < N; j += GSIZE) {
        const long long b = dbits(a[j]);
        if (bnan(b)) r.f = 1;
        else { const long long k = okey(b); r.mx = (k > r.mx) ? k : r.mx; r.mn = (k < r.mn) ? k : r.mn; }
    }
    r = group_reduce(r);
    nan = r.f;
    mx = unkey(r.mx); mn = unkey(r.mn);
}
// Python's builtin max() / min() of a numpy array (PyEMD check_imf: max(imf_old) - min(imf_old)).  Without NaN
// any order gives the same value; with NaN the result depends on positions, so one thread replays Python's loop.
__device__ void py_maxmin(const real_t* __restrict__ a, int N, real_t& mx, real_t& mn)
{
    int nan;
    group_maxmin(a, N, mx, mn, nan);
    if (nan) {                                          // uniform over the group
        if (GTID == 0) {
            real_t M = a[0], m = a[0];
            for (int j = 1; j < N; ++j) { const real_t v = a[j]; if (v > M) M = v; if (v < m) m = v; }
            CTL->stat[0] = M; CTL->stat[1] = m;
        }
        SYNC();
        mx = CTL->stat[0]; mn = CTL->stat[1];
        SYNC();
    }
}
// numpy's np.max / np.min (NaN propagates)
__device__ void np_maxmin(const real_t* __restrict__ a, int N, real_t& mx, real_t& mn)
{
    int nan;
    group_maxmin(a, N, mx, mn, nan);
    if (nan) { mx = __longlong_as_double(0x7ff8000000000000LL); mn = mx; }
}

// ============================================================================================================
//  numpy's np.sum of a 1-D float64 array, reproduced bit for bit: pairwise summation over blocks of at most 128
//  elements (8 interleaved accumulators, then ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), then the remainder in order),
//  combined along numpy's recursion tree (split at n/2 rounded down to a multiple of 8).  The tree depends only on N:
//  the host precomputes it (leaf starts and the combine operations level by level) -- `pw`.
//  The values of the summed terms come from a callable term(i, t[]) (K <= 3 sums at once).  Every sum this kernel
//  feeds into a decision (check_imf, end_condition) is therefore exactly numpy's, whatever the launch geometry.
// ============================================================================================================
struct PWS { int L, nlvl, root; const int* st; const int* lv; const int* ops; };
__device__ __forceinline__ PWS pws_load(const int* pw) {
    PWS s; s.L = pw[0]; s.nlvl = pw[1]; s.root = pw[2];
    s.st = pw + 3; s.lv = s.st + s.L + 1; s.ops = s.lv + s.nlvl + 1; return s;
}

template <int K, class F>
__device__ void pw_sum(const PWS& s, F term, real_t* out)
{
    const int p = BSH.par;
    real_t* P = BSH.pwP + (size_t)p * s.L * 24;
    real_t t[3];
    const int items = s.L * 8;
    for (int it = GTID; it < items; it += GSIZE) {          // lane j of leaf l: elements j, j+8, ... of its unrolled part
        const int leaf = it >> 3, j = it & 7;
        const int a = s.st[leaf], n = s.st[leaf + 1] - a;
        real_t acc[3];
        if (n < 8) {                                     // numpy: res = 0.; res += a[i] in order
            if (j != 0) continue;
            for (int k = 0; k < K; ++k) acc[k] = 0.0;
            for (int i = a; i < a + n; ++i) { term(i, t); for (int k = 0; k < K; ++k) acc[k] = acc[k] + t[k]; }
        } else {
            term(a + j, t);
            for (int k = 0; k < K; ++k) acc[k] = t[k];
            const int end8 = a + (n - (n & 7));
            for (int i = a + 8 + j; i < end8; i += 8) { term(i, t); for (int k = 0; k < K; ++k) acc[k] = acc[k] + t[k]; }
        }
        for (int k = 0; k < K; ++k) P[(size_t)it * 3 + k] = acc[k];
    }
    SYNC();
    real_t* node = BSH.pwN;                              // every block evaluates the (small) tree itself
    for (int leaf = threadIdx.x; leaf < s.L; leaf += blockDim.x) {
        const int a = s.st[leaf], n = s.st[leaf + 1] - a;
        const real_t* q = P + (size_t)leaf * 24;
        real_t res[3];
        if (n < 8) {
            for (int k = 0; k < K; ++k) res[k] = q[k];
        } else {
            for (int k = 0; k < K; ++k)
                res[k] = ((q[0 * 3 + k] + q[1 * 3 + k]) + (q[2 * 3 + k] + q[3 * 3 + k])) +
                         ((q[4 * 3 + k] + q[5 * 3 + k]) + (q[6 * 3 + k] + q[7 * 3 + k]));
            for (int i = a + (n - (n & 7)); i < a + n; ++i) { term(i, t); for (int k = 0; k < K; ++k) res[k] = res[k] + t[k]; }
        }
        for (int k = 0; k < K; ++k) node[(size_t)leaf * 3 + k] = res[k];
    }
    BLOCK_BARRIER();
    for (int l = 0; l < s.nlvl; ++l) {
        for (int op = s.lv[l] + (int)threadIdx.x; op < s.lv[l + 1]; op += blockDim.x) {
            const int d = s.ops[3 * op], x = s.ops[3 * op + 1], y = s.ops[3 * op + 2];
            for (int k = 0; k < K; ++k) node[(size_t)d * 3 + k] = node[(size_t)x * 3 + k] + node[(size_t)y * 3 + k];
        }
        BLOCK_BARRIER();
    }
    for (int k = 0; k < K; ++k) out[k] = node[(size_t)s.root * 3 + k];
    if (threadIdx.x == 0) BSH.par = p ^ 1;
    BLOCK_BARRIER();
}

// ============================================================================================================
//  Extrema detection, PyEMD "simple" semantics (EMD._find_extrema_simple), one contiguous chunk per group thread.
//
//  Events are produced in increasing position order inside a chunk and chunks are in thread order, so the
//  concatenation is the sorted list PyEMD builds with np.sort. Plateaus (runs of equal samples) belong to the
//  thread that owns the run's first sample; they never contain a regular extremum, so order is preserved.
//  Quirks reproduced on purpose: a plateau starting at d-index 1 or ending at the last d is dropped; a plateau
//  starting at d-index 0 uses d[-1] as its "before" slope; the midpoint is np.round (half to even).
//
//  Integer predicates: when three neighbouring samples are finite and not tiny (|x| >= 2^-463), the signs of
//  x*y, of (cur - prev) * (nxt - cur) and the equality tests are decided from the bit patterns: no product can
//  underflow to zero and no difference can vanish unless the operands are equal, so the result is the one the
//  floating-point expression gives -- but it costs integer instructions (full rate on every GPU) instead of
//  double-precision ones (1/32 or 1/64 rate on GeForce cards).  Any other sample takes the floating-point path.
// ============================================================================================================
struct ExtCounts { int cmax, cmin, cz; };
struct ExtState { int emax, emin, tmax, tmin, tz; };

__device__ __forceinline__ int plateau_event(const real_t* __restrict__ S, int N, int k, real_t prev, real_t cur,
                                             real_t nxt, int& epos, real_t& eval)
{
    int j = k + 1;
    real_t sj = nxt, sn = nxt;
    while (j <= N - 2) { sn = S[j + 1]; if (sn != sj) break; sj = sn; ++j; }
    const int fins = (j <= N - 2) ? j : (N - 1);
    if (k != 1 && fins != N - 1) {
        const real_t d_before = (k >= 1) ? (cur - prev) : (S[N - 1] - S[N - 2]);
        const real_t d_after = sn - sj;                       // d[fins] = S[fins+1] - S[fins]
        const int s2 = fins + k;
        const int q = s2 >> 1;
        epos = (s2 & 1) ? (q + (q & 1)) : q;                  // np.round half to even
        eval = cur;
        if (d_before > 0.0 && d_after < 0.0) return 1;
        if (d_before < 0.0 && d_after > 0.0) return 2;
    }
    return 0;
}

#if EXTREMA_MODE == 0
template <bool WRITE>
__device__ __forceinline__ ExtCounts extrema_walk(
    const real_t* __restrict__ S, int N, int lo, int hi,
    int off_max, int off_min, int tot_max, int tot_min,
    real_t* __restrict__ kx0, real_t* __restrict__ ky0, real_t* __restrict__ kx1, real_t* __restrict__ ky1)
{
    ExtCounts c; c.cmax = 0; c.cmin = 0; c.cz = 0;
    if (lo >= hi) return c;
    Ctl* C = CTL;
    real_t prev = (lo >= 1) ? S[lo - 1] : S[lo];
    real_t cur = S[lo];
    long long bp = dbits(prev), bc = dbits(cur);
    for (int k = lo; k < hi; ++k) {
        const bool has_next = (k + 1 < N);
        const real_t nxt = has_next ? S[k + 1] : cur;
        const long long bn = dbits(nxt);
        if (bz(bc) && (k == 0 || !bz(bp))) ++c.cz;                       // a run of exact zeros counts once
        if (has_next) {
            int ev = 0, epos = 0; real_t eval = 0.0;                      // ev: 1 maximum, 2 minimum
            if (bsafe(bp) && bsafe(bc) && bsafe(bn)) {                    // integer predicates (exact, see above)
                if ((bc ^ bn) < 0) ++c.cz;                                // cur * nxt < 0
                if (bn == bc && (k == 0 || bc != bp)) {                   // a plateau starts here
                    ev = plateau_event(S, N, k, prev, cur, nxt, epos, eval);
                } else if (k >= 1) {
                    const long long kp = okey(bp), kc = okey(bc), kn = okey(bn);
                    const int c1 = (kc > kp) - (kc < kp), c2 = (kn > kc) - (kn < kc);
                    if (c1 * c2 < 0) { ev = (c1 > 0) ? 1 : 2; epos = k; eval = cur; }
                }
            } else {                                                      // floating-point path (zeros, tiny, inf, NaN)
                if (cur * nxt < 0.0) ++c.cz;
                if (nxt == cur && (k == 0 || cur != prev)) {
                    ev = plateau_event(S, N, k, prev, cur, nxt, epos, eval);
                } else if (k >= 1) {
                    const real_t d1 = cur - prev, d2 = nxt - cur;
                    if (d1 * d2 < 0.0) { ev = (d1 > 0.0) ? 1 : 2; epos = k; eval = cur; }
                }
            }
            if (ev == 1) {
                if (WRITE) {
                    const int r = off_max + c.cmax;
                    kx0[R0 + r] = (real_t)epos; ky0[R0 + r] = eval;
                    if (r < FRONT) { C->fidx[0][r] = epos; C->fval[0][r] = eval; }
                    if (r >= tot_max - FRONT) { C->bidx[0][r - (tot_max - FRONT)] = epos; C->bval[0][r - (tot_max - FRONT)] = eval; }
                    if (eval < 0.0) atomicOr(&C->bad, 1);
                }
                ++c.cmax;
            } else if (ev == 2) {
                if (WRITE) {
                    const int r = off_min + c.cmin;
                    kx1[R0 + r] = (real_t)epos; ky1[R0 + r] = eval;
                    if (r < FRONT) { C->fidx[1][r] = epos; C->fval[1][r] = eval; }
                    if (r >= tot_min - FRONT) { C->bidx[1][r - (tot_min - FRONT)] = epos; C->bval[1][r - (tot_min - FRONT)] = eval; }
                    if (eval > 0.0) atomicOr(&C->bad, 1);
                }
                ++c.cmin;
            }
        }
        prev = cur; cur = nxt; bp = bc; bc = bn;
    }
    return c;
}

// Extrema pass, step 1 (all group threads): per-thread counts, group scan, totals (in the returned state).
__device__ ExtState extrema_count(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N) {
    const int T = GSIZE;
    const int m = (N + T - 1) / T;
    const int lo = min(GTID * m, N), hi = min(lo + m, N);
    ExtCounts c = extrema_walk<false>(S, N, lo, hi, 0, 0, 0, 0, (real_t*)0, (real_t*)0, (real_t*)0, (real_t*)0);
    ExtState es; int ez;
    group_scan3(c.cmax, c.cmin, c.cz, &es.emax, &es.emin, &ez, &es.tmax, &es.tmin, &es.tz);
    return es;
}

// Step 2 (all group threads): write the extrema to the knot arrays (slot R0 + rank), the first / last FRONT of each
// list to the group state, and set CTL->bad when a maximum is negative or a minimum positive.
__device__ void extrema_write(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N, const ExtState& es,
                              real_t* __restrict__ kx0, real_t* __restrict__ ky0,
                              real_t* __restrict__ kx1, real_t* __restrict__ ky1) {
    const int T = GSIZE;
    const int m = (N + T - 1) / T;
    const int lo = min(GTID * m, N), hi = min(lo + m, N);
    extrema_walk<true>(S, N, lo, hi, es.emax, es.emin, es.tmax, es.tmin, kx0, ky0, kx1, ky1);
    SYNC();
}
#endif   // EXTREMA_MODE == 0


// ============================================================================================================
//  Boundary mirroring, PyEMD EMD.prepare_points_simple, executed by ONE thread.  It only needs the first / last
//  FRONT extrema of each list (kept in the group state by the extrema write pass) plus S[0] and S[N-1].
//  Python slice semantics are reproduced exactly, including the "empty slice -> whole list" fallback and the two
//  "mirrored points outside the time range" fix-ups.  Duplicated positions are removed like np.delete does it
//  (the EARLIER one of two adjacent equal positions goes).  Mirrored knots are written to the slots left / right
//  of the real extrema, which the write pass put at slot R0 + rank.
// ============================================================================================================
#if EXTREMA_MODE == 0
struct SymList { int idx[NBSYM + 2]; real_t val[NBSYM + 2]; int n; };

__device__ __forceinline__ int    ext_idx(int e, int k, int tot) { return (k < FRONT) ? CTL->fidx[e][k] : CTL->bidx[e][k - (tot - FRONT)]; }
__device__ __forceinline__ real_t ext_val(int e, int k, int tot) { return (k < FRONT) ? CTL->fval[e][k] : CTL->bval[e][k - (tot - FRONT)]; }
__device__ __forceinline__ void sl_push(SymList& L, int idx, real_t val) { L.idx[L.n] = idx; L.val[L.n] = val; ++L.n; }
// appends ind_e[a:b][::-1]  (0 <= a, b <= tot)
__device__ __forceinline__ void sl_slice_rev(SymList& L, int e, int a, int b, int tot) {
    for (int k = b - 1; k >= a; --k) sl_push(L, ext_idx(e, k, tot), ext_val(e, k, tot));
}
// "if not arr.size: arr = ind_e" (only reachable when the list has a single element)
__device__ __forceinline__ void sl_fallback(SymList& L, int e, int tot) {
    if (L.n == 0) for (int k = 0; k < tot && k < NBSYM; ++k) sl_push(L, ext_idx(e, k, tot), ext_val(e, k, tot));
}

__device__ __forceinline__ void write_mirrored(int e, const SymList& Ls, int lsym, const SymList& Rs, int rsym, int tot,
                                               real_t* __restrict__ kx, real_t* __restrict__ ky)
{
    Ctl* C = CTL;
    const int first_real = C->fidx[e][0];
    bool keep[NBSYM + 2];
    int nk = 0;
    for (int j = 0; j < Ls.n; ++j) {
        const int pj = 2 * lsym - Ls.idx[j];
        const int pn = (j + 1 < Ls.n) ? (2 * lsym - Ls.idx[j + 1]) : first_real;
        keep[j] = (pj != pn);
        nk += keep[j] ? 1 : 0;
    }
    int slot = R0 - nk;
    for (int j = 0; j < Ls.n; ++j) if (keep[j]) {
        kx[slot] = (real_t)(2 * lsym - Ls.idx[j]); ky[slot] = Ls.val[j]; ++slot;
        if (e == 0 ? (Ls.val[j] < 0.0) : (Ls.val[j] > 0.0)) C->bad = 1;
    }
    C->kb[e] = R0 - nk;
    int nr = 0;
    for (int j = 0; j < Rs.n; ++j) {
        const int pj = 2 * rsym - Rs.idx[j];
        const int pn = (j + 1 < Rs.n) ? (2 * rsym - Rs.idx[j + 1]) : (-0x7fffffff);
        keep[j] = (pj != pn);
        nr += keep[j] ? 1 : 0;
    }
    slot = R0 + tot;
    for (int j = 0; j < Rs.n; ++j) if (keep[j]) {
        kx[slot] = (real_t)(2 * rsym - Rs.idx[j]); ky[slot] = Rs.val[j]; ++slot;
        if (e == 0 ? (Rs.val[j] < 0.0) : (Rs.val[j] > 0.0)) C->bad = 1;
    }
    C->kn[e] = nk + tot + nr;
}

__device__ void mirror_simple(const real_t* __restrict__ S, int N, int nmax, int nmin,
                              real_t* __restrict__ kx0, real_t* __restrict__ ky0,
                              real_t* __restrict__ kx1, real_t* __restrict__ ky1)
{
    const int nb = NBSYM;
    const real_t S0 = S[0], SL = S[N - 1];

    // ------------------------------------------------ left bound
    SymList lmax, lmin; lmax.n = 0; lmin.n = 0;
    int lsym;
    const int imax0 = ext_idx(0, 0, nmax), imin0 = ext_idx(1, 0, nmin);
    if (imax0 < imin0) {
        if (S0 > ext_val(1, 0, nmin)) {                                  // S[0] > S[ind_min[0]]
            sl_slice_rev(lmax, 0, 1, min(nmax, nb + 1), nmax);
            sl_slice_rev(lmin, 1, 0, min(nmin, nb), nmin);
            lsym = imax0;
        } else {
            sl_slice_rev(lmax, 0, 0, min(nmax, nb), nmax);
            sl_slice_rev(lmin, 1, 0, min(nmin, nb - 1), nmin); sl_push(lmin, 0, S0);
            lsym = 0;
        }
    } else {
        if (S0 < ext_val(0, 0, nmax)) {                                  // S[0] < S[ind_max[0]]
            sl_slice_rev(lmax, 0, 0, min(nmax, nb), nmax);
            sl_slice_rev(lmin, 1, 1, min(nmin, nb + 1), nmin);
            lsym = imin0;
        } else {
            sl_slice_rev(lmax, 0, 0, min(nmax, nb - 1), nmax); sl_push(lmax, 0, S0);
            sl_slice_rev(lmin, 1, 0, min(nmin, nb), nmin);
            lsym = 0;
        }
    }
    sl_fallback(lmin, 1, nmin);
    sl_fallback(lmax, 0, nmax);
    if (2 * lsym - lmin.idx[0] > 0 || 2 * lsym - lmax.idx[0] > 0) {      // mirrored points not outside the time range
        if (lsym == imax0) { lmax.n = 0; sl_slice_rev(lmax, 0, 0, min(nmax, nb), nmax); }
        else               { lmin.n = 0; sl_slice_rev(lmin, 1, 0, min(nmin, nb), nmin); }
        lsym = 0;
    }

    // ------------------------------------------------ right bound
    SymList rmax, rmin; rmax.n = 0; rmin.n = 0;
    int rsym;
    const int imaxL = ext_idx(0, nmax - 1, nmax), iminL = ext_idx(1, nmin - 1, nmin);
    if (imaxL < iminL) {
        if (SL < ext_val(0, nmax - 1, nmax)) {                           // S[-1] < S[ind_max[-1]]
            sl_slice_rev(rmax, 0, max(nmax - nb, 0), nmax, nmax);
            sl_slice_rev(rmin, 1, max(nmin - nb - 1, 0), nmin - 1, nmin);
            rsym = iminL;
        } else {
            sl_push(rmax, N - 1, SL); sl_slice_rev(rmax, 0, max(nmax - nb + 1, 0), nmax, nmax);
            sl_slice_rev(rmin, 1, max(nmin - nb, 0), nmin, nmin);
            rsym = N - 1;
        }
    } else {
        if (SL > ext_val(1, nmin - 1, nmin)) {                           // S[-1] > S[ind_min[-1]]
            sl_slice_rev(rmax, 0, max(nmax - nb - 1, 0), nmax - 1, nmax);
            sl_slice_rev(rmin, 1, max(nmin - nb, 0), nmin, nmin);
            rsym = imaxL;
        } else {
            sl_slice_rev(rmax, 0, max(nmax - nb, 0), nmax, nmax);
            sl_push(rmin, N - 1, SL); sl_slice_rev(rmin, 1, max(nmin - nb + 1, 0), nmin, nmin);
            rsym = N - 1;
        }
    }
    sl_fallback(rmin, 1, nmin);
    sl_fallback(rmax, 0, nmax);
    if (2 * rsym - rmin.idx[rmin.n - 1] < N - 1 || 2 * rsym - rmax.idx[rmax.n - 1] < N - 1) {
        if (rsym == imaxL) { rmax.n = 0; sl_slice_rev(rmax, 0, max(nmax - nb, 0), nmax, nmax); }
        else               { rmin.n = 0; sl_slice_rev(rmin, 1, max(nmin - nb, 0), nmin, nmin); }
        rsym = N - 1;
    }

    // (the real extrema were checked by the write pass; the mirrored values are checked in write_mirrored)
    write_mirrored(0, lmax, lsym, rmax, rsym, nmax, kx0, ky0);
    write_mirrored(1, lmin, lsym, rmin, rsym, nmin, kx1, ky1);
}
#endif   // EXTREMA_MODE == 0


// ============================================================================================================
//  PARABOL mode (extrema_detection="parabol"): EMD._find_extrema_parabol + EMD.prepare_points_parabol,
//  reproduced operation by operation (numpy evaluates every binary operation separately, left to right; the
//  build has fused multiply-add disabled, so each C++ operator below rounds exactly like the numpy ufunc).
//
//  Detection: samples that are interior points of a run of three equal values are dropped (np: _not_duplicate);
//  every remaining sample with both a previous and a next KEPT sample is the middle of a parabola
//  (Tp,Sp) (T0,S0) (Tn,Sn); its vertex counts when it lies in [T0 - (T0-Tp)/2, T0 + (Tn-T0)/2); a < 0 -> maximum,
//  a > 0 -> minimum.  Vertices of consecutive candidates are increasing, so the concatenation of the chunk results
//  of the threads is the list numpy builds.
//
//  Mirroring (prepare_points_parabol) has NO re-sorting / de-duplication: the extended knot lists are what the
//  Python code builds.  scipy then raises when they are not strictly increasing, and the envelopes only have the
//  length of the signal when the knots cover the whole position range -- check_knots flags both (PyEMD raises).
// ============================================================================================================
#if EXTREMA_MODE == 1
// np: _not_duplicate keeps the first and last sample and every interior sample that is not equal to both neighbours
__device__ __forceinline__ bool pb_kept(const real_t* __restrict__ S, int N, int k)
{
    return (k <= 0) || (k >= N - 1) || !(S[k] == S[k - 1] && S[k] == S[k + 1]);
}

// Vertex of the parabola through three kept samples; returns 1 (maximum), 2 (minimum) or 0 (no extremum counted).
__device__ __forceinline__ int pb_vertex(real_t Tp, real_t T0, real_t Tn, real_t Sp, real_t S0, real_t Sn,
                                         real_t& pos, real_t& val)
{
    const real_t TnTp = Tn - Tp, T0Tn = T0 - Tn, TpT0 = Tp - T0;
    const real_t scale = Tp * Tn * Tn + Tp * Tp * T0 + T0 * T0 * Tn - Tp * Tp * Tn - Tp * T0 * T0 - T0 * Tn * Tn;
    real_t a = T0Tn * Sp + TnTp * S0 + TpT0 * Sn;
    real_t b = (S0 - Sn) * (Tp * Tp) + (Sn - Sp) * (T0 * T0) + (Sp - S0) * (Tn * Tn);
    real_t c = T0 * Tn * T0Tn * Sp + Tn * Tp * TnTp * S0 + Tp * T0 * TpT0 * Sn;
    a = a / scale;
    b = b / scale;
    c = c / scale;
    if (a == 0.0) a = 1e-14;
    const real_t tv = -0.5 * b / a;
    if (!((tv < T0 + 0.5 * (Tn - T0)) && (tv >= T0 - 0.5 * (T0 - Tp)))) return 0;
    const real_t sv = a * tv * tv + b * tv + c;
    pos = tv; val = sv;
    return (a < 0.0) ? 1 : ((a > 0.0) ? 2 : 0);
}

// Candidates whose middle sample lies in [lo, hi).  Counts maxima / minima; with WRITE also stores them at slot
// R0 + rank (kx / ky), keeps the first / last FRONT of each list in the group state and sets CTL->bad.
template <bool WRITE>
__device__ __forceinline__ void pb_walk(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N, int lo, int hi,
                                        int off_max, int off_min, int tot_max, int tot_min,
                                        real_t* __restrict__ kx0, real_t* __restrict__ ky0,
                                        real_t* __restrict__ kx1, real_t* __restrict__ ky1, int& cmax, int& cmin)
{
    cmax = 0; cmin = 0;
    Ctl* C = CTL;
    int ci = (lo > 1) ? lo : 1;
    while (ci < hi && ci <= N - 2 && !pb_kept(S, N, ci)) ++ci;
    if (ci >= hi || ci > N - 2) return;
    int pi = ci - 1;
    while (pi > 0 && !pb_kept(S, N, pi)) --pi;
    real_t Tp = TT[pi], Sp = S[pi], T0 = TT[ci], S0 = S[ci];
    while (true) {
        int ni = ci + 1;
        while (ni < N - 1 && !pb_kept(S, N, ni)) ++ni;
        const real_t Tn = TT[ni], Sn = S[ni];
        real_t pos = 0.0, val = 0.0;
        const int kind = pb_vertex(Tp, T0, Tn, Sp, S0, Sn, pos, val);
        if (kind == 1) {
            if (WRITE) {
                const int r = off_max + cmax;
                kx0[R0 + r] = pos; ky0[R0 + r] = val;
                if (r < FRONT) { C->fpos[0][r] = pos; C->fval[0][r] = val; }
                if (r >= tot_max - FRONT) { C->bpos[0][r - (tot_max - FRONT)] = pos; C->bval[0][r - (tot_max - FRONT)] = val; }
                if (val < 0.0) atomicOr(&C->bad, 1);
            }
            ++cmax;
        } else if (kind == 2) {
            if (WRITE) {
                const int r = off_min + cmin;
                kx1[R0 + r] = pos; ky1[R0 + r] = val;
                if (r < FRONT) { C->fpos[1][r] = pos; C->fval[1][r] = val; }
                if (r >= tot_min - FRONT) { C->bpos[1][r - (tot_min - FRONT)] = pos; C->bval[1][r - (tot_min - FRONT)] = val; }
                if (val > 0.0) atomicOr(&C->bad, 1);
            }
            ++cmin;
        }
        Tp = T0; Sp = S0; T0 = Tn; S0 = Sn; ci = ni;
        if (ci >= hi || ci > N - 2) break;
    }
}

// zero crossings (indzer) of samples [lo, hi): strict sign changes S[k]*S[k+1] < 0, plus one per run of exact zeros
__device__ __forceinline__ int pb_zero_count(const real_t* __restrict__ S, int N, int lo, int hi)
{
    int cz = 0;
    if (lo >= hi) return 0;
    real_t prev = (lo >= 1) ? S[lo - 1] : S[lo];
    real_t cur = S[lo];
    long long bp = dbits(prev), bc = dbits(cur);
    for (int k = lo; k < hi; ++k) {
        const bool has_next = (k + 1 < N);
        const real_t nxt = has_next ? S[k + 1] : cur;
        const long long bn = dbits(nxt);
        if (bz(bc) && (k == 0 || !bz(bp))) ++cz;
        if (has_next) {
            if (bsafe(bc) && bsafe(bn)) { if ((bc ^ bn) < 0) ++cz; }
            else if (cur * nxt < 0.0) ++cz;
        }
        prev = cur; cur = nxt; bp = bc; bc = bn;
    }
    return cz;
}

__device__ ExtState extrema_count(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N)
{
    const int T = GSIZE;
    const int m = (N + T - 1) / T;
    const int lo = min(GTID * m, N), hi = min(lo + m, N);
    int cmax, cmin;
    pb_walk<false>(S, TT, N, lo, hi, 0, 0, 0, 0, (real_t*)0, (real_t*)0, (real_t*)0, (real_t*)0, cmax, cmin);
    const int cz = pb_zero_count(S, N, lo, hi);
    ExtState es; int ez;
    group_scan3(cmax, cmin, cz, &es.emax, &es.emin, &ez, &es.tmax, &es.tmin, &es.tz);
    return es;
}

__device__ void extrema_write(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N, const ExtState& es,
                              real_t* __restrict__ kx0, real_t* __restrict__ ky0,
                              real_t* __restrict__ kx1, real_t* __restrict__ ky1)
{
    const int T = GSIZE;
    const int m = (N + T - 1) / T;
    const int lo = min(GTID * m, N), hi = min(lo + m, N);
    int cmax, cmin;
    pb_walk<true>(S, TT, N, lo, hi, es.emax, es.emin, es.tmax, es.tmin, kx0, ky0, kx1, ky1, cmax, cmin);
    SYNC();
}

// ---- mirroring (ONE thread) -----------------------------------------------------------------------------------
struct PList { real_t pos[NBSYM + 2]; real_t val[NBSYM + 2]; int n; };

__device__ __forceinline__ real_t pext_pos(int e, int k, int tot) { return (k < FRONT) ? CTL->fpos[e][k] : CTL->bpos[e][k - (tot - FRONT)]; }
__device__ __forceinline__ real_t pext_val(int e, int k, int tot) { return (k < FRONT) ? CTL->fval[e][k] : CTL->bval[e][k - (tot - FRONT)]; }
__device__ __forceinline__ void pl_push(PList& L, real_t pos, real_t val) { L.pos[L.n] = pos; L.val[L.n] = val; ++L.n; }
// appends  (c2 - pos[a:b]) / val[a:b]  of list e (Python slice semantics: b is clipped to the list length)
__device__ __forceinline__ void pl_reflect(PList& L, int e, int a, int b, int tot, real_t c2)
{
    if (b > tot) b = tot;
    for (int k = a; k < b; ++k) pl_push(L, c2 - pext_pos(e, k, tot), pext_val(e, k, tot));
}

// writes  expand_left[::-1], extrema, expand_right[::-1]  of one list: the left part into the slots before R0, the
// right part behind the real extrema; records the knot range in the group state and checks the mirrored values.
__device__ __forceinline__ void pwrite_mirrored(int e, const PList& Ls, const PList& Rs, int tot,
                                                real_t* __restrict__ kx, real_t* __restrict__ ky)
{
    Ctl* C = CTL;
    const int nl = Ls.n, nr = Rs.n;
    for (int j = 0; j < nl; ++j) {
        const real_t p = Ls.pos[nl - 1 - j], v = Ls.val[nl - 1 - j];
        kx[R0 - nl + j] = p; ky[R0 - nl + j] = v;
        if (e == 0 ? (v < 0.0) : (v > 0.0)) C->bad = 1;
    }
    for (int j = 0; j < nr; ++j) {
        const real_t p = Rs.pos[nr - 1 - j], v = Rs.val[nr - 1 - j];
        kx[R0 + tot + j] = p; ky[R0 + tot + j] = v;
        if (e == 0 ? (v < 0.0) : (v > 0.0)) C->bad = 1;
    }
    C->kb[e] = R0 - nl;
    C->kn[e] = nl + tot + nr;
}

__device__ void mirror_points(const real_t* __restrict__ S, const real_t* __restrict__ TT, int N, int nmax, int nmin,
                              real_t* __restrict__ kx0, real_t* __restrict__ ky0,
                              real_t* __restrict__ kx1, real_t* __restrict__ ky1)
{
    const int nb = NBSYM;
    const real_t S0 = S[0], SL = S[N - 1], T0 = TT[0], TL = TT[N - 1];
    PList lmax, lmin, rmax, rmin;
    lmax.n = 0; lmin.n = 0; rmax.n = 0; rmin.n = 0;

    // ------------------------------------------------ left bound
    const real_t pmax0 = pext_pos(0, 0, nmax), pmin0 = pext_pos(1, 0, nmin);
    const real_t dl = pmax0 - pmin0;
    if (dl < 0.0) {                                                         // left extremum is a maximum
        if ((S0 > pext_val(1, 0, nmin)) && (fabs(dl) > (pmax0 - T0))) {     // mirror signal to first extremum
            const real_t c2 = 2.0 * pmax0;
            pl_reflect(lmax, 0, 1, nb + 1, nmax, c2);
            pl_reflect(lmin, 1, 0, nb, nmin, c2);
        } else {                                                            // mirror signal to beginning
            const real_t c2 = 2.0 * T0;
            pl_reflect(lmax, 0, 0, nb, nmax, c2);
            pl_push(lmin, c2 - T0, S0);
            pl_reflect(lmin, 1, 0, nb - 1, nmin, c2);
        }
    } else {                                                                // left extremum is a minimum
        if ((S0 < pext_val(0, 0, nmax)) && (fabs(dl) > (pmin0 - T0))) {
            const real_t c2 = 2.0 * pmin0;
            pl_reflect(lmax, 0, 0, nb, nmax, c2);
            pl_reflect(lmin, 1, 1, nb + 1, nmin, c2);
        } else {
            const real_t c2 = 2.0 * T0;
            pl_push(lmax, c2 - T0, S0);
            pl_reflect(lmax, 0, 0, nb - 1, nmax, c2);
            pl_reflect(lmin, 1, 0, nb, nmin, c2);
        }
    }

    // ------------------------------------------------ right bound
    const real_t pmaxL = pext_pos(0, nmax - 1, nmax), pminL = pext_pos(1, nmin - 1, nmin);
    const real_t dr = pmaxL - pminL;
    if (!(dr > 0.0)) {                                                      // last extremum is a minimum
        if ((SL < pext_val(0, nmax - 1, nmax)) && (fabs(dr) > (TL - pminL))) {
            const int ia = max(0, nmax - nb), ib = max(0, nmin - nb - 1);
            const real_t c2 = 2.0 * pminL;
            pl_reflect(rmax, 0, ia, nmax, nmax, c2);
            pl_reflect(rmin, 1, ib, nmin - 1, nmin, c2);
        } else {
            const int ia = max(0, nmax - nb + 1), ib = max(0, nmin - nb);
            const real_t c2 = 2.0 * TL;
            pl_reflect(rmax, 0, ia, nmax, nmax, c2);
            pl_push(rmax, c2 - TL, SL);
            pl_reflect(rmin, 1, ib, nmin, nmin, c2);
        }
    } else {                                                                // last extremum is a maximum
        if ((SL > pext_val(1, nmin - 1, nmin)) && (nmax > 1) && (fabs(dr) > (TL - pmaxL))) {
            const int ia = max(0, nmax - nb - 1), ib = max(0, nmin - nb);
            const real_t c2 = 2.0 * pmaxL;
            pl_reflect(rmax, 0, ia, nmax - 1, nmax, c2);
            pl_reflect(rmin, 1, ib, nmin, nmin, c2);
        } else {
            const int ia = max(0, nmax - nb), ib = max(0, nmin - nb + 1);
            const real_t c2 = 2.0 * TL;
            pl_reflect(rmax, 0, ia, nmax, nmax, c2);
            pl_reflect(rmin, 1, ib, nmin, nmin, c2);
            pl_push(rmin, c2 - TL, SL);
        }
    }

    pwrite_mirrored(0, lmax, rmax, nmax, kx0, ky0);
    pwrite_mirrored(1, lmin, rmin, nmin, kx1, ky1);
}

#endif   // EXTREMA_MODE == 1


// ============================================================================================================
//  Cubic spline stage:  scipy.interpolate.CubicSpline(x, y)  (not-a-knot ends), reproduced bit for bit.
//
//  scipy builds a tridiagonal system (build rows D, U, L, b), solves it with LAPACK dgtsv (Gaussian elimination
//  with partial pivoting restricted to adjacent rows), then converts the derivatives s_i to piecewise
//  polynomial coefficients (CubicHermiteSpline formulas).  dgtsv is a first-order recurrence along the rows
//  whose state is (d, u, b) per row, so it cannot be split exactly by cyclic reduction (which re-associates the
//  arithmetic and has no pivoting).  Instead -- a speculative chunked recurrence with an exactness certificate:
//
//    * every group thread owns a chunk of rows; it first "warms up" from a fixed number of rows earlier (starting from the raw,
//      unmodified row), which makes the entry state of the chunk equal to the exact one in all but rare cases (the
//      recurrence contracts by roughly a factor 4 per row), then eliminates its own rows and stores the records;
//    * the exit state of chunk c is compared bit for bit with the entry state chunk c+1 used.  If every pair
//      matches, every chunk is exact by induction from the first one (whose entry is the true initial state).
//      Chunks whose entry disagrees are re-solved, all at once, from their neighbour's exit state, until no pair
//      disagrees (parallel fix-up rounds; normally none is needed);
//    * back substitution runs the same way (two-term recurrence from the end, warm-up from the right).
//
//  The result is identical to dgtsv, not merely close.
//
//  Coefficients are generated per piece when the evaluation walk enters it:
//      t = ((s0 + s1) - 2*slope)/dx ;  c = [t/dx, (slope - s0)/dx - t, s0, y0]
//      value = ((c3 + c2*s) + c1*(s*s)) + c0*((s*s)*s),  s = x - x_left          (PPoly.__call__ order)
// ============================================================================================================
struct EnvBuf {
    real_t* kx;      // knot positions, first knot at index 0
    real_t* ky;      // knot values
    real_t* sv;      // solved derivatives s_i (cubic, pchip, akima, cubic_hermite) / parameters (three point) / coefficients (quadratic)
    real_t* rec;     // per-row records of the solver, REC_W doubles per row (cubic: D, U, L, B;  quadratic: a row of the banded LU)
    int n;           // knot count
    int mode;        // 0 cubic spline (n >= 4) | 1 three-point cubic (n == 3) | -1 unsupported (PyEMD raises)
};

#if SPLINE_KIND == 0     // the speculative chunked tridiagonal solver: scipy's CubicSpline only
struct FSt { real_t d, u, b; };

template <bool WRITE>
__device__ __forceinline__ void fwd_step(int i, int n, const real_t* __restrict__ X, const real_t* __restrict__ Y,
                                         real_t* __restrict__ rec, FSt& st, real_t& dxc, real_t& slc, int& bad)
{
    real_t Dn, Un, bn, li, ndx = 0.0, nsl = 0.0;
    if (i + 1 < n - 1) {                                    // row i+1 is an interior row
        ndx = X[i + 2] - X[i + 1];
        nsl = (Y[i + 2] - Y[i + 1]) / ndx;
        Dn = 2.0 * (dxc + ndx); Un = dxc; bn = 3.0 * (ndx * slc + dxc * nsl); li = ndx;
        bad |= !(ndx > 0.0);
    } else {                                                // i == n-2: row i+1 is the last (not-a-knot) row
        const real_t dxp = X[n - 2] - X[n - 3];
        const real_t slp = (Y[n - 2] - Y[n - 3]) / dxp;
        const real_t d = X[n - 1] - X[n - 3];
        Dn = dxp; Un = 0.0; li = d;                         // D[n-1] = dx[n-3] (scipy: A[1, -1] = dx[-2])
        bn = ((dxc * dxc) * slp + ((2.0 * d + dxc) * dxp) * slc) / d;
    }
    real_t o0, o1, o2, o3;
    if (fabs(st.d) >= fabs(li)) {                           // no row interchange
        const real_t fact = li / st.d;
        o0 = st.d; o1 = st.u; o2 = 0.0; o3 = st.b;
        const real_t nd = Dn - fact * st.u, nb = bn - fact * st.b;
        st.d = nd; st.u = Un; st.b = nb;
    } else {                                                // interchange rows i and i+1
        const real_t fact = st.d / li;
        o0 = li; o1 = Dn; o2 = Un; o3 = bn;
        const real_t nd = st.u - fact * Dn, nu = -fact * Un, nb = st.b - fact * bn;
        st.d = nd; st.u = nu; st.b = nb;
    }
    if (WRITE) {
        double2* r = reinterpret_cast<double2*>(rec + 4 * (size_t)i);
        r[0] = make_double2(o0, o1);
        r[1] = make_double2(o2, o3);
    }
    dxc = ndx; slc = nsl;
}

// Forward elimination of rows [a, end).  entry_in (optional): the exact state of row a; otherwise it is estimated by
// a warm-up from the raw row a - warm.  vq (optional): receives entry state (3) and exit state (3).
__device__ __forceinline__ void fwd_chunk(const EnvBuf& E, int a, int end, real_t* __restrict__ vq,
                                          const real_t* __restrict__ entry_in, int warm)
{
    const real_t* __restrict__ X = E.kx; const real_t* __restrict__ Y = E.ky;
    real_t* __restrict__ rec = E.rec;
    const int n = E.n;
    FSt st; real_t dxc = 0.0, slc = 0.0; int bad = 0;
    int i;
    if (entry_in) {                                         // exact entry state supplied (repair)
        st.d = entry_in[0]; st.u = entry_in[1]; st.b = entry_in[2];
        if (a <= n - 2) { dxc = X[a + 1] - X[a]; slc = (Y[a + 1] - Y[a]) / dxc; }
        i = a;
    } else {
        const int st0 = (a > warm) ? (a - warm) : 0;
        if (st0 == 0) {                                     // true first row of the system
            const real_t dx0 = X[1] - X[0], dx1 = X[2] - X[1];
            const real_t sl0 = (Y[1] - Y[0]) / dx0, sl1 = (Y[2] - Y[1]) / dx1;
            const real_t d = X[2] - X[0];
            st.d = dx1; st.u = d;
            st.b = (((dx0 + 2.0 * d) * dx1) * sl0 + (dx0 * dx0) * sl1) / d;
            dxc = dx0; slc = sl0;
            bad |= (!(dx0 > 0.0)) | (!(dx1 > 0.0));
        } else {                                            // raw interior row st0 (1 <= st0 <= n-2)
            const real_t dxm = X[st0] - X[st0 - 1];
            const real_t slm = (Y[st0] - Y[st0 - 1]) / dxm;
            dxc = X[st0 + 1] - X[st0];
            slc = (Y[st0 + 1] - Y[st0]) / dxc;
            st.d = 2.0 * (dxm + dxc); st.u = dxm; st.b = 3.0 * (dxc * slm + dxm * slc);
        }
        i = st0;
        for (; i < a; ++i) fwd_step<false>(i, n, X, Y, rec, st, dxc, slc, bad);
    }
    if (vq) { vq[0] = st.d; vq[1] = st.u; vq[2] = st.b; }
    for (; i < end; ++i) {
        if (i < n - 1) {
            fwd_step<true>(i, n, X, Y, rec, st, dxc, slc, bad);
        } else {                                            // last row: nothing left to eliminate
            double2* r = reinterpret_cast<double2*>(rec + 4 * (size_t)i);
            r[0] = make_double2(st.d, 0.0);
            r[1] = make_double2(0.0, st.b);
        }
    }
    if (vq) { vq[3] = st.d; vq[4] = st.u; vq[5] = st.b; }
    if (bad) CTL->degenerate = 1;
}

// Back substitution of rows [a, end): x_i = ((B_i - U_i x_{i+1}) - L_i x_{i+2}) / D_i, x_n = x_{n+1} = 0.
// entry_in (optional): the exact (x_end, x_{end+1}); otherwise estimated by a warm-up from the right.
// vq (optional): receives the entry values used.
__device__ __forceinline__ void bwd_chunk(const EnvBuf& E, int a, int end, real_t* __restrict__ vq,
                                          const real_t* __restrict__ entry_in, int warm)
{
    const real_t* __restrict__ rec = E.rec;
    real_t* __restrict__ sv = E.sv;
    const int n = E.n;
    real_t x1 = 0.0, x2 = 0.0;
    if (entry_in) {
        x1 = entry_in[0]; x2 = entry_in[1];
    } else {
        const int top = (end + warm < n) ? (end + warm) : n;
        for (int i = top - 1; i >= end; --i) {
            const double2 r0 = *reinterpret_cast<const double2*>(rec + 4 * (size_t)i);
            const double2 r1 = *reinterpret_cast<const double2*>(rec + 4 * (size_t)i + 2);
            const real_t xi = ((r1.y - r0.y * x1) - r1.x * x2) / r0.x;
            x2 = x1; x1 = xi;
        }
    }
    if (vq) { vq[6] = x1; vq[7] = x2; }
    for (int i = end - 1; i >= a; --i) {
        const double2 r0 = *reinterpret_cast<const double2*>(rec + 4 * (size_t)i);
        const double2 r1 = *reinterpret_cast<const double2*>(rec + 4 * (size_t)i + 2);
        const real_t xi = ((r1.y - r0.y * x1) - r1.x * x2) / r0.x;
        sv[i] = xi;
        x2 = x1; x1 = xi;
    }
}

// All group threads.  Solves the not-a-knot cubic systems of the envelopes in cubic mode (mode == 0).
// vs: per-group verification scratch, VS_DOUBLES per group thread.
//
// Exactness: every chunk records the entry state it started from.  A chunk whose entry differs from its left neighbour's
// exit state is re-solved from that exit state, in parallel with every other such chunk, and the check repeats until a
// round re-solves nothing.  The first chunk starts from the true initial state, so after round k at least the first k+1
// chunks are exact; the loop ends exactly when every chunk's entry is its neighbour's exit, i.e. when the whole
// solution equals the sequential recurrence's.  With the warm-up lengths used (forward 36 rows, backward 44 rows --
// the contraction of the recurrence makes the warmed-up state bit-identical) the first round normally finds nothing.
__device__ void cubic_solve(const EnvBuf& E0, const EnvBuf& E1, real_t* __restrict__ vs)
{
    const int T = GSIZE;
    const int n0 = (E0.mode == 0) ? E0.n : 0, n1 = (E1.mode == 0) ? E1.n : 0;
    if (n0 + n1 == 0) return;                               // uniform over the group
    int m = (n0 + n1 + (T - 3)) / (T - 2);                  // ceil((n0 + n1) / (T - 2)): at most T chunks overall
    if (m < BSH.minchunk) m = BSH.minchunk;
    const int nch0 = (n0 + m - 1) / m, nch1 = (n1 + m - 1) / m;
    const int q = GTID;
    int e = 0, c = q;
    bool active = true;
    if (q >= nch0) { e = 1; c = q - nch0; active = (q < nch0 + nch1); }
    const EnvBuf E = (e == 0) ? E0 : E1;
    const int a = c * m;
    const int end = (a + m < E.n) ? (a + m) : E.n;
    real_t* vq = vs + (size_t)q * VS_DOUBLES;
    if (q == 0) { CTL->rep[0] = 0; CTL->rep[1] = 0; }
    if (active) fwd_chunk(E, a, end, vq, (const real_t*)0, BSH.warm_f);   // forward elimination (warm-up + own rows)
    for (int r = 0; ; ++r) {                                 // forward fix-up rounds
        SYNC();
        bool need = false;
        real_t ex[3] = { 0.0, 0.0, 0.0 };
        if (active && c > 0) {
            const real_t* pv = vq - VS_DOUBLES;
            ex[0] = pv[3]; ex[1] = pv[4]; ex[2] = pv[5];
            need = !(ex[0] == vq[0] && ex[1] == vq[1] && ex[2] == vq[2]);
        }
        SYNC();
        if (need) { fwd_chunk(E, a, end, vq, ex, BSH.warm_f); atomicOr(&CTL->rep[r & 1], 1); }
        SYNC();
        const int any = CTL->rep[r & 1];
        if (q == 0) CTL->rep[(r + 1) & 1] = 0;
        if (!any) break;
    }
    if (active) bwd_chunk(E, a, end, vq, (const real_t*)0, BSH.warm_b);   // chunked back substitution
    for (int r = 0; ; ++r) {                                 // backward fix-up rounds
        SYNC();
        bool need = false;
        real_t ex[3] = { 0.0, 0.0, 0.0 };
        if (active && end < E.n) {
            ex[0] = E.sv[end]; ex[1] = (end + 1 < E.n) ? E.sv[end + 1] : 0.0;
            need = !(vq[6] == ex[0] && vq[7] == ex[1]);
        }
        SYNC();
        if (need) { bwd_chunk(E, a, end, vq, ex, BSH.warm_b); atomicOr(&CTL->rep[r & 1], 1); }
        SYNC();
        const int any = CTL->rep[r & 1];
        if (q == 0) CTL->rep[(r + 1) & 1] = 0;
        if (!any) break;
    }
}
#endif                   // SPLINE_KIND == 0 (cubic solver)

// ---- piece coefficients / evaluation ---------------------------------------------------------------------------
struct Piece {
    real_t c0, c1, c2, c3, xl, xr;
#if INTPOS
    int xli, xri;                       // integer knot positions (simple mode): exact integer compares / differences
#endif
#if SPLINE_KIND == 3
    real_t e0, e1, e2;                  // quadratic B-spline: the three active coefficients
#endif
};

// index of the last piece (pieces are the knot intervals; the quadratic B-spline has two fewer: its knots are midpoints)
__device__ __forceinline__ int piece_last(const EnvBuf& E)
{
#if SPLINE_KIND == 3
    return E.n - 3;
#else
    return E.n - 2;
#endif
}

#if SPLINE_KIND == 0 || SPLINE_KIND >= 4       // cubic polynomial pieces: CubicSpline and the Hermite family (shared PPoly conversion)
__device__ __forceinline__ void piece_cubic(const EnvBuf& E, int i, Piece& p)
{
    const real_t xl = E.kx[i], xr = E.kx[i + 1], yl = E.ky[i], yr = E.ky[i + 1];
    const real_t s0 = E.sv[i], s1 = E.sv[i + 1];
    const real_t dx = xr - xl;
    const real_t slope = (yr - yl) / dx;
    const real_t t = ((s0 + s1) - 2.0 * slope) / dx;
    p.c0 = t / dx; p.c1 = (slope - s0) / dx - t; p.c2 = s0; p.c3 = yl;
    p.xl = xl; p.xr = xr;
}

__device__ __forceinline__ real_t eval_cubic_s(const Piece& p, real_t s)          // s = x - x_left
{
    const real_t s2 = s * s;
    return ((p.c3 + p.c2 * s) + p.c1 * s2) + p.c0 * (s2 * s);
}
#endif

#if SPLINE_KIND == 3
// knot vector of scipy's interpolating quadratic B-spline (make_interp_spline k=2): the data points' midpoints without
// the first and last one, the end points repeated three times:  t[i], 0 <= i <= n+2
__device__ __forceinline__ real_t qt(const EnvBuf& E, int i)
{
    if (i <= 2) return E.kx[0];
    if (i >= E.n) return E.kx[E.n - 1];
    return (E.kx[i - 1] + E.kx[i - 2]) / 2.0;
}
__device__ __forceinline__ real_t piece_left(const EnvBuf& E, int i) { return qt(E, i + 2); }
#else
__device__ __forceinline__ real_t piece_left(const EnvBuf& E, int i) { return E.kx[i]; }
#endif

// Piece containing x: the largest i in [0, last] with left(i) <= x (binary search; any thread).
__device__ __forceinline__ int find_piece(const EnvBuf& E, real_t x)
{
    int lo = 0, hi = piece_last(E);
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (piece_left(E, mid) <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

#if USE_START_TABLE
// All group threads: fills the start table entry k of envelope e with the piece containing the first sample (k * ms)
// of sample chunk k.  (simple mode only: positions are the integers 0..N-1, the pieces' boundaries are knots)
__device__ __forceinline__ void start_table(const EnvBuf& E, int e, int ms, int nsc)
{
    int* tab = BSH.siv + (size_t)e * BSH.sivs;
    for (int i = GTID; i < E.n - 1; i += GSIZE) {
        const int xl = (int)E.kx[i], xr = (int)E.kx[i + 1];
        const int k0 = (xl <= 0) ? 0 : (xl + ms - 1) / ms;
        int k1 = (i == E.n - 2) ? (nsc - 1) : ((xr <= 0) ? -1 : ((xr + ms - 1) / ms - 1));
        if (k1 > nsc - 1) k1 = nsc - 1;
        for (int k = k0; k <= k1; ++k) tab[k] = i;
    }
}
#endif

#if SPLINE_KIND == 0
// ---- three-knot envelopes: PyEMD.splines.cubic_spline_3pts -------------------------------------------------------
// numpy solves the 3x3 system with LAPACK dgesv from its OpenBLAS: left-looking LU, pivot reciprocal scaling,
// column-oriented substitution; the dot products / axpys are fused multiply-adds or not depending on which OpenBLAS
// kernel the CPU selects (P3_FMA, probed on the host against the installed numpy).  The matrix is column diagonally
// dominant, so partial pivoting never exchanges rows.  Stores a1, b1, a2, b2 in E.sv[0..3].  ONE thread.
__device__ void three_point_setup(const EnvBuf& E)
{
    const real_t x0 = E.kx[0], x1 = E.kx[1], x2 = E.kx[2], y0 = E.ky[0], y1 = E.ky[1], y2 = E.ky[2];
    const real_t x1x0 = x1 - x0, x2x1 = x2 - x1, y1y0 = y1 - y0, y2y1 = y2 - y1;
    const real_t ia = 1.0 / x1x0, ib = 1.0 / x2x1;
    const real_t v1 = ((3.0 * y1y0) * ia) * ia, v3 = ((3.0 * y2y1) * ib) * ib, v2 = v1 + v3;
    const real_t a00 = 2.0 * ia, a01 = ia, a02 = 0.0;
    real_t a10 = ia, a11 = 2.0 * (ia + ib), a12 = ib;
    real_t a20 = 0.0, a21 = ib, a22 = 2.0 * ib;
    const real_t r0 = 1.0 / a00;
    a10 = a10 * r0; a20 = a20 * r0;
    a11 = a11 - P3F(a10, a01, 0.0);
    a21 = a21 - P3F(a20, a01, 0.0);
    const real_t r1 = 1.0 / a11;
    a21 = a21 * r1;
    a12 = a12 - P3F(a10, a02, 0.0);
    { real_t s = P3F(a20, a02, 0.0); s = P3F(a21, a12, s); a22 = a22 - s; }
    real_t b0 = v1, b1 = v2, b2 = v3;
    b1 = P3F(-b0, a10, b1); b2 = P3F(-b0, a20, b2); b2 = P3F(-b1, a21, b2);
    b2 = b2 / a22; b0 = P3F(-b2, a02, b0); b1 = P3F(-b2, a12, b1);
    b1 = b1 / a11; b0 = P3F(-b1, a01, b0);
    b0 = b0 / a00;
    E.sv[0] = b0 * x1x0 - y1y0;
    E.sv[1] = (-b1) * x1x0 + y1y0;
    E.sv[2] = b1 * x2x1 - y2y1;
    E.sv[3] = (-b2) * x2x1 + y2y1;
}

__device__ __forceinline__ void piece_3pt(const EnvBuf& E, int i, Piece& p)
{
    p.xl = E.kx[i]; p.xr = E.kx[i + 1]; p.c2 = E.ky[i]; p.c3 = E.ky[i + 1];
    p.c0 = E.sv[2 * i]; p.c1 = E.sv[2 * i + 1];
}

__device__ __forceinline__ real_t eval_3pt_t(const Piece& p, real_t t1)            // t1 = (x - xl) / (xr - xl)
{
    const real_t t11 = 1.0 - t1;
    return ((t11 * p.c2) + (t1 * p.c3)) + ((t1 * t11) * ((p.c0 * t11) + (p.c1 * t1)));
}
#endif   // SPLINE_KIND == 0 (three-knot envelopes)

#if SPLINE_KIND == 1
// interp1d(kind="linear") -> np.interp:  slope*(x - xl) + yl, the last knot returns its own value
__device__ __forceinline__ void piece_lin(const EnvBuf& E, int i, Piece& p)
{
    p.xl = E.kx[i]; p.xr = E.kx[i + 1]; p.c2 = E.ky[i]; p.c3 = E.ky[i + 1];
    p.c0 = (p.c3 - p.c2) / (p.xr - p.xl);
}
__device__ __forceinline__ real_t eval_lin(const Piece& p, real_t x)
{
    if (x >= p.xr) return p.c3;
    if (x == p.xl) return p.c2;
    return p.c0 * (x - p.xl) + p.c2;
}
#endif
#if SPLINE_KIND == 2
// interp1d(kind="slinear") -> make_interp_spline(k=1) = BSpline with t = [x0, x..., x_last], evaluated by de Boor:
//   w = 1/(xb - xa); h0 = 0 + w*(xb - x); h1 = w*(x - xa); value = (0 + y_i*h0) + y_{i+1}*h1
__device__ __forceinline__ void piece_slin(const EnvBuf& E, int i, Piece& p)
{
    p.xl = E.kx[i]; p.xr = E.kx[i + 1]; p.c2 = E.ky[i]; p.c3 = E.ky[i + 1];
    p.c0 = (p.xr == p.xl) ? 0.0 : (1.0 / (p.xr - p.xl));
}
__device__ __forceinline__ real_t eval_slin(const Piece& p, real_t x)
{
    real_t h0 = 0.0, h1 = 0.0;
    if (p.xr != p.xl) { h0 = 0.0 + p.c0 * (p.xr - x); h1 = p.c0 * (x - p.xl); }
    return (0.0 + p.c2 * h0) + p.c3 * h1;
}
#endif
#if SPLINE_KIND == 3
// interp1d(kind="quadratic") -> make_interp_spline(k=2): coefficients c (E.sv) from quad_setup, evaluated by de Boor.
// Piece i is the knot interval l = i + 2 of qt(): [t[l], t[l+1]).
__device__ __forceinline__ void piece_quad(const EnvBuf& E, int i, Piece& p)
{
    const int l = i + 2;
    p.c0 = qt(E, l - 1); p.xl = qt(E, l); p.xr = qt(E, l + 1); p.c1 = qt(E, l + 2);
    p.e0 = E.sv[l - 2]; p.e1 = E.sv[l - 1]; p.e2 = E.sv[l];
}
// the three non-zero quadratic B-splines at x on the interval with knots (tm1, t0, t1, t2) -- scipy's _deBoor_D, m = 0
__device__ __forceinline__ void quad_basis(real_t tm1, real_t t0, real_t t1, real_t t2, real_t x,
                                           real_t& h0, real_t& h1, real_t& h2)
{
    real_t a0, a1;
    // j = 1
    if (t1 == t0) { a0 = 0.0; a1 = 0.0; }
    else { const real_t w = 1.0 / (t1 - t0); a0 = 0.0 + w * (t1 - x); a1 = w * (x - t0); }
    // j = 2
    const real_t g0 = a0, g1 = a1;
    h0 = 0.0;
    if (t1 == tm1) { h1 = 0.0; }
    else { const real_t w = g0 / (t1 - tm1); h0 = h0 + w * (t1 - x); h1 = w * (x - tm1); }
    if (t2 == t0) { h2 = 0.0; }
    else { const real_t w = g1 / (t2 - t0); h1 = h1 + w * (t2 - x); h2 = w * (x - t0); }
}
__device__ __forceinline__ real_t eval_quad(const Piece& p, real_t x)
{
    real_t h0, h1, h2;
    quad_basis(p.c0, p.xl, p.xr, p.c1, x, h0, h1, h2);
    return ((0.0 + p.e0 * h0) + p.e1 * h1) + p.e2 * h2;
}
#endif

__device__ __forceinline__ void piece_load(const EnvBuf& E, int i, Piece& p)
{
#if SPLINE_KIND == 1
    piece_lin(E, i, p);
#elif SPLINE_KIND == 2
    piece_slin(E, i, p);
#elif SPLINE_KIND == 3
    piece_quad(E, i, p);
#elif SPLINE_KIND == 0
    if (E.mode == 0) piece_cubic(E, i, p); else piece_3pt(E, i, p);
#else                                   // pchip, akima, cubic_hermite: always a Hermite cubic piece
    piece_cubic(E, i, p);
#endif
#if INTPOS
    p.xli = (int)p.xl; p.xri = (int)p.xr;
#endif
}

// value of the envelope at position x (floating-point positions: parabol mode, quadratic pieces)
__device__ __forceinline__ real_t piece_eval(int mode, const Piece& p, real_t x)
{
#if SPLINE_KIND == 1
    return eval_lin(p, x);
#elif SPLINE_KIND == 2
    return eval_slin(p, x);
#elif SPLINE_KIND == 3
    return eval_quad(p, x);
#elif SPLINE_KIND == 0
    return (mode == 0) ? eval_cubic_s(p, x - p.xl) : eval_3pt_t(p, (x - p.xl) / (p.xr - p.xl));
#else
    return eval_cubic_s(p, x - p.xl);
#endif
}

#if INTPOS
// value at the integer position j (simple mode): x - xl and xr - x are exact integers, formed without double ops
__device__ __forceinline__ real_t piece_eval_i(int mode, const Piece& p, int j)
{
#if SPLINE_KIND == 1
    if (j >= p.xri) return p.c3;
    if (j == p.xli) return p.c2;
    return p.c0 * (real_t)(j - p.xli) + p.c2;
#elif SPLINE_KIND == 2
    real_t h0 = 0.0, h1 = 0.0;
    if (p.xri != p.xli) { h0 = 0.0 + p.c0 * (real_t)(p.xri - j); h1 = p.c0 * (real_t)(j - p.xli); }
    return (0.0 + p.c2 * h0) + p.c3 * h1;
#elif SPLINE_KIND == 0
    return (mode == 0) ? eval_cubic_s(p, (real_t)(j - p.xli))
                       : eval_3pt_t(p, (real_t)(j - p.xli) / (real_t)(p.xri - p.xli));
#else
    return eval_cubic_s(p, (real_t)(j - p.xli));
#endif
}
#endif


// ---- envelope kind -------------------------------------------------------------------------------------------------
// mode: 0 piecewise cubic Hermite (derivatives in E.sv) | 1 PyEMD's three-point cubic | -1 PyEMD / scipy would raise
__device__ __forceinline__ int env_mode(int n)
{
#if SPLINE_KIND == 0
    return (n >= 4) ? 0 : ((n == 3) ? 1 : -1);
#elif SPLINE_KIND == 3
    return (n >= 3) ? 0 : -1;               // interp1d(kind="quadratic") needs three points
#else
    return (n >= 2) ? 0 : -1;
#endif
}

#if SPLINE_KIND == 4 || SPLINE_KIND == 5 || SPLINE_KIND == 6
// ---- Hermite-family derivative generators (all rows independent -> trivially parallel) ----------------------------------
__device__ __forceinline__ bool sign_differs(real_t a, real_t b)
{
    if (a != a || b != b) return true;                      // np.sign(nan) = nan, and nan != x
    return ((a > 0.0) - (a < 0.0)) != ((b > 0.0) - (b < 0.0));
}

#if SPLINE_KIND == 4
// np.gradient(Y, X): central three-point formula in the interior (a uniform-spacing shortcut when every spacing is
// equal, as numpy decides), one-sided differences at the ends.
__device__ void hermite_setup(const EnvBuf& E0, const EnvBuf& E1)
{
    long long c0 = 0, c1 = 0;
    const real_t d00 = E0.kx[1] - E0.kx[0], d01 = E1.kx[1] - E1.kx[0];
    for (int i = GTID; i < E0.n - 1; i += GSIZE) if ((E0.kx[i + 1] - E0.kx[i]) != d00) c0 += 1;
    for (int i = GTID; i < E1.n - 1; i += GSIZE) if ((E1.kx[i + 1] - E1.kx[i]) != d01) c1 += 1;
    GRed r = gred_id(); r.s = c0; r.s2 = c1;
    r = group_reduce(r);
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const bool uniform = (e == 0) ? (r.s == 0) : (r.s2 == 0);
        const real_t d0 = (e == 0) ? d00 : d01;
        const int n = E.n;
        for (int i = GTID; i < n; i += GSIZE) {
            real_t s;
            if (i == 0) {
                s = (E.ky[1] - E.ky[0]) / d0;
            } else if (i == n - 1) {
                s = (E.ky[n - 1] - E.ky[n - 2]) / (E.kx[n - 1] - E.kx[n - 2]);
            } else if (uniform) {
                s = (E.ky[i + 1] - E.ky[i - 1]) / (2.0 * d0);
            } else {
                const real_t dx1 = E.kx[i] - E.kx[i - 1], dx2 = E.kx[i + 1] - E.kx[i];
                const real_t a = -(dx2) / (dx1 * (dx1 + dx2));
                const real_t b = (dx2 - dx1) / (dx1 * dx2);
                const real_t c = dx1 / (dx2 * (dx1 + dx2));
                s = (a * E.ky[i - 1] + b * E.ky[i]) + c * E.ky[i + 1];
            }
            E.sv[i] = s;
        }
    }
    SYNC();
}
#endif

#if SPLINE_KIND == 5
// scipy PchipInterpolator._find_derivatives / _edge_case
__device__ __forceinline__ real_t pchip_edge(real_t h0, real_t h1, real_t m0, real_t m1)
{
    real_t d = (((2.0 * h0 + h1) * m0) - (h0 * m1)) / (h0 + h1);
    if (sign_differs(d, m0)) return 0.0;
    if (sign_differs(m0, m1) && (fabs(d) > 3.0 * fabs(m0))) return 3.0 * m0;
    return d;
}

__device__ void hermite_setup(const EnvBuf& E0, const EnvBuf& E1)
{
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const int n = E.n;
        for (int i = GTID; i < n; i += GSIZE) {
            real_t s;
            if (n == 2) {
                s = (E.ky[1] - E.ky[0]) / (E.kx[1] - E.kx[0]);
            } else if (i == 0) {
                const real_t h0 = E.kx[1] - E.kx[0], h1 = E.kx[2] - E.kx[1];
                s = pchip_edge(h0, h1, (E.ky[1] - E.ky[0]) / h0, (E.ky[2] - E.ky[1]) / h1);
            } else if (i == n - 1) {
                const real_t h0 = E.kx[n - 1] - E.kx[n - 2], h1 = E.kx[n - 2] - E.kx[n - 3];
                s = pchip_edge(h0, h1, (E.ky[n - 1] - E.ky[n - 2]) / h0, (E.ky[n - 2] - E.ky[n - 3]) / h1);
            } else {
                const real_t hp = E.kx[i] - E.kx[i - 1], hn = E.kx[i + 1] - E.kx[i];
                const real_t mp = (E.ky[i] - E.ky[i - 1]) / hp, mn = (E.ky[i + 1] - E.ky[i]) / hn;
                const bool cond = sign_differs(mn, mp) || (mn == 0.0) || (mp == 0.0);
                if (cond) {
                    s = 0.0;
                } else {
                    const real_t w1 = 2.0 * hn + hp, w2 = hn + 2.0 * hp;
                    const real_t whmean = (w1 / mp + w2 / mn) / (w1 + w2);
                    s = 1.0 / whmean;
                }
            }
            E.sv[i] = s;
        }
    }
    SYNC();
}
#endif

#if SPLINE_KIND == 6
// scipy Akima1DInterpolator (method="akima"): slopes m[0..n+2] (two extrapolated at each end), derivative = weighted
// mean of neighbouring slopes unless the weights vanish (relative to the largest weight sum -- xp.max, NaN propagates
// -- cutoff 1e-9).  E.rec is used as the m array (n + 3 doubles).
__device__ void hermite_setup(const EnvBuf& E0, const EnvBuf& E1)
{
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const int n = E.n;
        if (n == 2) continue;
        for (int k = GTID + 2; k <= n; k += GSIZE)
            E.rec[k] = (E.ky[k - 1] - E.ky[k - 2]) / (E.kx[k - 1] - E.kx[k - 2]);
    }
    SYNC();
    if (GTID == 0) {
        for (int e = 0; e < 2; ++e) {
            const EnvBuf& E = (e == 0) ? E0 : E1;
            const int n = E.n;
            if (n == 2) continue;
            real_t* m = E.rec;
            m[1] = 2.0 * m[2] - m[3];
            m[0] = 2.0 * m[1] - m[2];
            m[n + 1] = 2.0 * m[n] - m[n - 1];
            m[n + 2] = 2.0 * m[n + 1] - m[n];
        }
    }
    SYNC();
    GRed r = gred_id();                                     // mx: max key of envelope 0's f12; mn: min key of -f12 of envelope 1
    for (int i = GTID; i < E0.n && E0.n > 2; i += GSIZE) {
        const real_t* m = E0.rec;
        const long long b = dbits(fabs(m[i + 3] - m[i + 2]) + fabs(m[i + 1] - m[i]));
        if (bnan(b)) r.f |= 1; else { const long long k = okey(b); r.mx = (k > r.mx) ? k : r.mx; }
    }
    for (int i = GTID; i < E1.n && E1.n > 2; i += GSIZE) {
        const real_t* m = E1.rec;
        const long long b = dbits(-(fabs(m[i + 3] - m[i + 2]) + fabs(m[i + 1] - m[i])));
        if (bnan(b)) r.f |= 2; else { const long long k = okey(b); r.mn = (k < r.mn) ? k : r.mn; }
    }
    r = group_reduce(r);
    const real_t qnan = __longlong_as_double(0x7ff8000000000000LL);
    const real_t t0 = (r.f & 1) ? qnan : unkey(r.mx);
    const real_t t1 = (r.f & 2) ? qnan : -unkey(r.mn);
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const int n = E.n;
        const real_t thr = 1.0e-9 * ((e == 0) ? t0 : t1);
        for (int i = GTID; i < n; i += GSIZE) {
            real_t s;
            if (n == 2) {
                s = (E.ky[1] - E.ky[0]) / (E.kx[1] - E.kx[0]);
            } else {
                const real_t* m = E.rec;
                const real_t f1 = fabs(m[i + 3] - m[i + 2]), f2 = fabs(m[i + 1] - m[i]);
                const real_t f12 = f1 + f2;
#if AKIMA_FORM == 0
                if (f12 > thr) s = ((f1 * m[i + 1]) + (f2 * m[i + 2])) / f12;                 // scipy <= 1.16
#else
                if (f12 > thr) s = m[i + 1] + ((f2 / f12) * (m[i + 2] - m[i + 1]));           // scipy >= 1.17
#endif
                else s = 0.5 * (m[i + 3] + m[i]);
            }
            E.sv[i] = s;
        }
    }
    SYNC();
}
#endif
#endif   // SPLINE_KIND 4, 5, 6

#if SPLINE_KIND == 3
// ============================================================================================================
//  Quadratic spline: interp1d(kind="quadratic") = make_interp_spline(x, y, k=2), reproduced bit for bit.
//
//  scipy builds the collocation matrix (row i: the three quadratic B-splines that are non-zero at x_i, found by de Boor's
//  recursion; band matrix with kl = ku = 2) and solves it with LAPACK dgbsv:  dgbtf2 (right-looking LU with partial
//  pivoting; multiplier scaling by the reciprocal of the pivot, rank-one update by DGER) and dgbtrs (the L solve replays
//  the interchanges and rank-one updates on the right hand side; the U solve is DTBSV: a TRUE division by the diagonal,
//  then an axpy of the column above it).  Whether DGER / DAXPY fuse their multiply-adds depends on the OpenBLAS kernel
//  the CPU selects (QUAD_FMA, probed on the host against the installed scipy).  The collocation rows are independent;
//  the LU is a recurrence along the rows, run by one thread per envelope as a streaming window of three rows x five
//  columns (band plus fill-in), with the L solve fused into it.  The rows of U (5 doubles per row) go to E.rec.
// ============================================================================================================
__device__ __forceinline__ void quad_row(const real_t* __restrict__ rec, int i, int jref,
                                         real_t& r0, real_t& r1, real_t& r2, real_t& r3, real_t& r4, int& bad)
{
    const real_t* r = rec + (size_t)REC_W * i;
    const real_t h0 = r[0], h1 = r[1], h2 = r[2];
    const int o = (int)r[3] - jref;                         // first column of the row relative to the window's column
    r0 = 0.0; r1 = 0.0; r2 = 0.0; r3 = 0.0; r4 = 0.0;
    if (o == 0)      { r0 = h0; r1 = h1; r2 = h2; }
    else if (o == 1) { r1 = h0; r2 = h1; r3 = h2; }
    else if (o == 2) { r2 = h0; r3 = h1; r4 = h2; }
    else bad = 1;                                           // outside the band: cannot happen for increasing knots
}

#define Q_SWAP(x, y) do { const real_t t_ = (x); (x) = (y); (y) = t_; } while (0)

// ONE thread.  rec: collocation rows in, rows of U out; b (E.sv): right hand side in, coefficients out.
// Returns 1 when the matrix is singular (LAPACK info > 0: scipy raises LinAlgError).
__device__ int quad_factor_solve(const EnvBuf& E)
{
    const int n = E.n;
    real_t* __restrict__ rec = E.rec;
    real_t* __restrict__ b = E.sv;
    real_t a0, a1, a2, a3, a4, p0, p1, p2, p3, p4, q0, q1, q2, q3, q4;
    int bad = 0;
    quad_row(rec, 0, 0, a0, a1, a2, a3, a4, bad);           // window: rows j, j+1, j+2 as columns j .. j+4
    quad_row(rec, 1, 0, p0, p1, p2, p3, p4, bad);
    quad_row(rec, 2, 0, q0, q1, q2, q3, q4, bad);
    if (bad) return 1;
    int ju = 0;                                             // last column touched so far (dgbtf2's JU)
    for (int j = 0; j < n; ++j) {
        const int km = (n - 1 - j < 2) ? (n - 1 - j) : 2;
        int jp = 0;                                         // idamax: first maximum of |column j| over rows j .. j+km
        real_t pm = fabs(a0);
        if (km >= 1) { const real_t v = fabs(p0); if (v > pm) { jp = 1; pm = v; } }
        if (km >= 2) { const real_t v = fabs(q0); if (v > pm) { jp = 2; pm = v; } }
        if (pm == 0.0) return 1;
        { const int t = j + 2 + jp; const int c = (t < n - 1) ? t : (n - 1); if (c > ju) ju = c; }
        if (jp == 1) {
            Q_SWAP(a0, p0); Q_SWAP(a1, p1); Q_SWAP(a2, p2); Q_SWAP(a3, p3); Q_SWAP(a4, p4);
            Q_SWAP(b[j], b[j + 1]);
        } else if (jp == 2) {
            Q_SWAP(a0, q0); Q_SWAP(a1, q1); Q_SWAP(a2, q2); Q_SWAP(a3, q3); Q_SWAP(a4, q4);
            Q_SWAP(b[j], b[j + 2]);
        }
        if (km > 0) {
            const int uo = ju - j;                          // columns j+1 .. ju take part in the rank-one update
            const real_t r = 1.0 / a0;
            p0 = p0 * r;
            if (km == 2) q0 = q0 * r;
            if (uo >= 1) { const real_t y0 = -1.0 * a1; p1 = QF(y0, p0, p1); if (km == 2) q1 = QF(y0, q0, q1); }
            if (uo >= 2) { const real_t y0 = -1.0 * a2; p2 = QF(y0, p0, p2); if (km == 2) q2 = QF(y0, q0, q2); }
            if (uo >= 3) { const real_t y0 = -1.0 * a3; p3 = QF(y0, p0, p3); if (km == 2) q3 = QF(y0, q0, q3); }
            if (uo >= 4) { const real_t y0 = -1.0 * a4; p4 = QF(y0, p0, p4); if (km == 2) q4 = QF(y0, q0, q4); }
            const real_t yb = -1.0 * b[j];                  // dgbtrs, L solve: the same step on the right hand side
            b[j + 1] = QF(yb, p0, b[j + 1]);
            if (km == 2) b[j + 2] = QF(yb, q0, b[j + 2]);
        }
        real_t* u = rec + (size_t)REC_W * j;                // row j of U
        u[0] = a0; u[1] = a1; u[2] = a2; u[3] = a3; u[4] = a4;
        a0 = p1; a1 = p2; a2 = p3; a3 = p4; a4 = 0.0;       // slide the window by one column / row
        p0 = q1; p1 = q2; p2 = q3; p3 = q4; p4 = 0.0;
        if (j + 3 < n) quad_row(rec, j + 3, j + 1, q0, q1, q2, q3, q4, bad);
        else { q0 = 0.0; q1 = 0.0; q2 = 0.0; q3 = 0.0; q4 = 0.0; }
        if (bad) return 1;
    }
    for (int jj = n - 1; jj >= 0; --jj) {                   // dtbsv (upper, 4 super diagonals): divide, then axpy upwards
        const real_t v = b[jj] / rec[(size_t)REC_W * jj];
        b[jj] = v;
        const real_t da = -v;
        const int len = (jj < 4) ? jj : 4;
        for (int t = 1; t <= len; ++t) { const int q = jj - t; b[q] = QF(da, rec[(size_t)REC_W * q + t], b[q]); }
    }
    return 0;
}

// All group threads: collocation rows of both envelopes, then the serial solves (envelope 0 on group thread 0,
// envelope 1 on the first thread of the second warp, so that the two chains run side by side).  Result in E.sv.
__device__ void quad_setup(const EnvBuf& E0, const EnvBuf& E1)
{
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        for (int i = GTID; i < E.n; i += GSIZE) {
            const real_t x = E.kx[i];
            const int pc = find_piece(E, x);                // scipy's interval l = pc + 2
            const int l = pc + 2;
            real_t h0, h1, h2;
            quad_basis(qt(E, l - 1), qt(E, l), qt(E, l + 1), qt(E, l + 2), x, h0, h1, h2);
            real_t* r = E.rec + (size_t)REC_W * i;
            r[0] = h0; r[1] = h1; r[2] = h2; r[3] = (real_t)pc; r[4] = 0.0;
            E.sv[i] = E.ky[i];
        }
    }
    SYNC();
    const int t1 = (blockDim.x >= 64) ? 32 : 1;
    if (GTID == 0 || GTID == t1) {
        const EnvBuf& E = (GTID == 0) ? E0 : E1;
        if (quad_factor_solve(E) != 0) atomicOr(&CTL->degenerate, 3);
    }
    SYNC();
}
#endif   // SPLINE_KIND == 3

// All group threads: builds whatever the envelopes need before evaluation (solves / derivatives, 3-point parameters).
__device__ void spline_setup(const EnvBuf& E0, const EnvBuf& E1, real_t* __restrict__ vs)
{
#if SPLINE_KIND == 0
    if (E0.mode == 0 || E1.mode == 0) cubic_solve(E0, E1, vs);
    if (GTID == 0) {
        if (E0.mode == 1) three_point_setup(E0);
        if (E1.mode == 1) three_point_setup(E1);
    }
    SYNC();
#elif SPLINE_KIND == 4 || SPLINE_KIND == 5 || SPLINE_KIND == 6
    hermite_setup(E0, E1);
#elif SPLINE_KIND == 3
    quad_setup(E0, E1);
#endif                                                        // linear / slinear: the pieces come straight from the knots
}

#if EXTREMA_MODE == 1
// All group threads, after mirroring.  scipy's CubicSpline / PCHIP / Akima / CubicHermite raise unless the knots are
// strictly increasing, and PyEMD only gets envelopes of the signal's length when the knots span all positions
// (t = T[(T >= x_first) & (T <= x_last)]).  degenerate = 3 marks "PyEMD raises here".
__device__ void check_knots(const EnvBuf& E0, const EnvBuf& E1, const real_t* __restrict__ TT, int N, bool need_increasing)
{
    if (GTID == 0) {
        if (!(E0.kx[0] <= TT[0]) || !(E0.kx[E0.n - 1] >= TT[N - 1]) ||
            !(E1.kx[0] <= TT[0]) || !(E1.kx[E1.n - 1] >= TT[N - 1])) atomicOr(&CTL->degenerate, 3);
    }
    if (need_increasing) {
        for (int i = GTID; i < E0.n - 1; i += GSIZE) if (!(E0.kx[i + 1] > E0.kx[i])) atomicOr(&CTL->degenerate, 3);
        for (int i = GTID; i < E1.n - 1; i += GSIZE) if (!(E1.kx[i + 1] > E1.kx[i])) atomicOr(&CTL->degenerate, 3);
    }
    SYNC();
}
#if KNOTS_STRICT == 0
// interp1d sorts its knots (np.argsort(kind="mergesort"), a STABLE sort) before it builds anything.  All group threads,
// after the coverage check.  Knots already in order (the normal case) cost one pass; otherwise every knot's rank is
// (number of knots with a smaller position) + (number of EARLIER knots with an equal position).
__device__ void sort_knots(const EnvBuf& E0, const EnvBuf& E1)
{
    if (CTL->degenerate != 0) return;                       // uniform over the group (set before check_knots' barrier)
    if (GTID == 0) CTL->unsorted = 0;
    SYNC();
    for (int i = GTID; i < E0.n - 1; i += GSIZE) if (E0.kx[i + 1] < E0.kx[i]) atomicOr(&CTL->unsorted, 1);
    for (int i = GTID; i < E1.n - 1; i += GSIZE) if (E1.kx[i + 1] < E1.kx[i]) atomicOr(&CTL->unsorted, 1);
    SYNC();
    if (CTL->unsorted == 0) return;
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const int n = E.n;
        real_t* tx = E.rec;                                 // temporary sorted copies (rec has room for 2n doubles)
        real_t* ty = E.rec + n;
        for (int i = GTID; i < n; i += GSIZE) {
            const real_t v = E.kx[i];
            int r = 0;
            for (int j = 0; j < n; ++j) { const real_t w = E.kx[j]; r += ((w < v) || (w == v && j < i)) ? 1 : 0; }
            tx[r] = v; ty[r] = E.ky[i];
        }
    }
    SYNC();
    for (int e = 0; e < 2; ++e) {
        const EnvBuf& E = (e == 0) ? E0 : E1;
        const int n = E.n;
        for (int i = GTID; i < n; i += GSIZE) { E.kx[i] = E.rec[i]; E.ky[i] = E.rec[n + i]; }
    }
    SYNC();
}

#if SPLINE_KIND == 2 || SPLINE_KIND == 3
// make_interp_spline raises "Expect x to not have duplicates" (sorted knots: equal neighbours).
__device__ void check_dups(const EnvBuf& E0, const EnvBuf& E1)
{
    for (int i = GTID; i < E0.n - 1; i += GSIZE) if (!(E0.kx[i + 1] > E0.kx[i])) atomicOr(&CTL->degenerate, 3);
    for (int i = GTID; i < E1.n - 1; i += GSIZE) if (!(E1.kx[i + 1] > E1.kx[i])) atomicOr(&CTL->degenerate, 3);
    SYNC();
}
#endif
#endif   // KNOTS_STRICT == 0

#endif   // EXTREMA_MODE == 1


// ============================================================================================================
//  Per-sample passes.
// ============================================================================================================
// One sifting update (all group threads):  mean = 0.5*(max_env + min_env);  new = old - mean.   Each thread owns a
// contiguous chunk of samples and walks the pieces of both envelopes (the first piece comes from the start table).
// No statistics here: check_imf's sums are formed only when PyEMD would look at them (see the main loop).
__device__ void sift_update(const real_t* __restrict__ cur, real_t* __restrict__ nxt, const real_t* __restrict__ TT, int N,
                            const EnvBuf& E0, const EnvBuf& E1)
{
    const int T = GSIZE;
    const int ms = (N + T - 1) / T;
#if USE_START_TABLE
    const int nsc = (N + ms - 1) / ms;
    start_table(E0, 0, ms, nsc);
    start_table(E1, 1, ms, nsc);
    SYNC();
#endif
    const int gt = GTID;
    const int lo = gt * ms, hi = min(lo + ms, N);
    const int L0 = piece_last(E0), L1 = piece_last(E1);
    if (lo < N) {
#if USE_START_TABLE
        int i0 = BSH.siv[gt], i1 = BSH.siv[(size_t)BSH.sivs + gt];
#else
        int i0 = find_piece(E0, XPOS(TT, lo)), i1 = find_piece(E1, XPOS(TT, lo));
#endif
        i0 = (i0 < 0) ? 0 : ((i0 > L0) ? L0 : i0);
        i1 = (i1 < 0) ? 0 : ((i1 > L1) ? L1 : i1);
        Piece p0, p1;
        piece_load(E0, i0, p0);
        piece_load(E1, i1, p1);
        for (int j = lo; j < hi; ++j) {
#if INTPOS
            while (i0 < L0 && j >= p0.xri) { ++i0; piece_load(E0, i0, p0); }
            while (i1 < L1 && j >= p1.xri) { ++i1; piece_load(E1, i1, p1); }
            const real_t a = piece_eval_i(E0.mode, p0, j);
            const real_t b = piece_eval_i(E1.mode, p1, j);
#else
            const real_t x = XPOS(TT, j);
            while (i0 < L0 && x >= p0.xr) { ++i0; piece_load(E0, i0, p0); }
            while (i1 < L1 && x >= p1.xr) { ++i1; piece_load(E1, i1, p1); }
            const real_t a = piece_eval(E0.mode, p0, x);
            const real_t b = piece_eval(E1.mode, p1, x);
#endif
            const real_t mean = 0.5 * (a + b);
            nxt[j] = cur[j] - mean;
        }
    }
    SYNC();
    if (GTID == 0) CTL->bad = 0;                           // every thread has read this iteration's flag; reset for the next
}


// ============================================================================================================
//  Main kernel: G thread blocks per signal; reproduces PyEMD.EMD.emd (extrema, nbsym mirroring, spline, check_imf,
//  end_condition) value for value and decision for decision.
//
//  Scratch, per signal (doubles):  bufA bufB acc [N4 each] | envelope 0, 1 [ENV_KC * kc each] | vs [VS * GSIZE] |
//  pairwise lane partials [2 * L * 24] | start table [GSIZE ints] | exchange slots [2 * G * 8 words] |
//  G per-block tree-node arrays.  The offsets come from the host (lay[]).  ctl: per signal, zeroed by the host:
//  barrier words, then the group state (used when G > 1).
// ============================================================================================================
#define DBG(nv, ev, zv) do { if (GTID == 0 && imf_count < max_imfs) { \
    debug_n_out[(size_t)b * max_imfs + imf_count] = (nv); debug_extno_out[(size_t)b * max_imfs + imf_count] = (ev); \
    debug_nzm_out[(size_t)b * max_imfs + imf_count] = (zv); } } while (0)

__device__ void group_setup(int G, real_t* __restrict__ scratch, const long long* __restrict__ lay,
                            int* __restrict__ ctl_base, int ctl_words, int warm, int minchunk)   // warm = fwd | bwd << 16
{
    const int b = blockIdx.x / G, gb = blockIdx.x % G;
    if (threadIdx.x == 0) {
        BSH.G = G; BSH.gb = gb; BSH.gs = G * (int)blockDim.x; BSH.gt0 = gb * (int)blockDim.x; BSH.par = 0;
        BSH.warm_f = warm & 0xffff; BSH.warm_b = (warm >> 16) & 0xffff; BSH.minchunk = minchunk;
        real_t* sg = scratch + (size_t)b * (size_t)lay[0];
        BSH.gx = reinterpret_cast<long long*>(sg + lay[5]);
        BSH.pwP = sg + lay[3];
        BSH.pwN = sg + lay[6] + (size_t)gb * (size_t)lay[7];
        int* cb = ctl_base + (size_t)b * ctl_words;
        BSH.bar = reinterpret_cast<unsigned*>(cb);
        if (G == 1) {
            BSH.c = &BSH.ctl;
            BSH.siv = &BSH.start_iv[0][0]; BSH.sivs = THREADS;
            BSH.ctl.bad = 0; BSH.ctl.degenerate = 0; BSH.ctl.unsorted = 0; BSH.ctl.err = 0; BSH.ctl.nanf = 0;
        } else {
            BSH.c = reinterpret_cast<Ctl*>(cb + 4);         // zeroed by the host
            BSH.siv = reinterpret_cast<int*>(sg + lay[4]); BSH.sivs = BSH.gs;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(THREADS) void emd_batch_kernel(
    const real_t* __restrict__ signals, const real_t* __restrict__ TT, int N, int max_imfs, int G,
    real_t std_thr, real_t svar_thr, real_t energy_ratio_thr, real_t total_power_thr, real_t range_thr,
    int max_iteration, int fixe, int fixe_h, int max_imf, int warm, int minchunk,
    real_t* __restrict__ imfs_out, int* __restrict__ n_imfs_out, int* __restrict__ err_out,
    int* __restrict__ exhausted_out, int* __restrict__ imf_count_out, int* __restrict__ maxit_out,
    real_t* __restrict__ scratch, const long long* __restrict__ lay, int kcap, const int* __restrict__ pw,
    int* __restrict__ ctl_base, int ctl_words,
    int* __restrict__ debug_n_out, int* __restrict__ debug_extno_out, int* __restrict__ debug_nzm_out)
{
    group_setup(G, scratch, lay, ctl_base, ctl_words, warm, minchunk);
    const int b = blockIdx.x / G;
    const int N4 = ALIGN4(N), kc = ALIGN4(kcap);
    const real_t* S = signals + (size_t)b * N;
    real_t* sc = scratch + (size_t)b * (size_t)lay[0];
    real_t* bufA = sc;
    real_t* bufB = sc + N4;
    real_t* acc = sc + 2 * (size_t)N4;
    real_t* ek = sc + lay[1];
    real_t* kx0 = ek;               real_t* ky0 = ek + kc;               real_t* sv0 = ek + 2 * (size_t)kc;   real_t* rc0 = ek + 3 * (size_t)kc;
    real_t* kx1 = ek + ENV_KC * (size_t)kc; real_t* ky1 = kx1 + kc;      real_t* sv1 = kx1 + 2 * (size_t)kc;  real_t* rc1 = kx1 + 3 * (size_t)kc;
    real_t* vs = sc + lay[2];
    real_t* out_base = imfs_out + (size_t)b * max_imfs * N;
    const int kmax = kc - R0 - NBSYM - 2;       // most extrema one list can hold
    const PWS pws = pws_load(pw);

    for (int j = GTID; j < N; j += GSIZE) bufA[j] = S[j];
    SYNC();

    real_t* cur = bufA;     // the imf being sifted (initially the residue = S)
    real_t* nxt = bufB;
    int imf_count = 0, n_maxit = 0;
    bool finished = false;
    int extNo_last = -1;    // PyEMD's `extNo`, which persists across IMFs (initially -1)

    while (!finished && imf_count < max_imfs - 1) {
        int n = 0, n_h = 0;
        bool have_counts = false;
        ExtState es; es.emax = 0; es.emin = 0; es.tmax = 0; es.tmin = 0; es.tz = 0;
        while (true) {
            ++n;
            if (n >= max_iteration) {                       // PyEMD: log, accept the IMF as it stands, move on
                ++n_maxit;
                DBG(n, -1, -1);
                break;
            }
            if (!have_counts) es = extrema_count(cur, TT, N);
            have_counts = false;
            const int nmax = es.tmax, nmin = es.tmin, nzm = es.tz;
            const int extNo = nmax + nmin;
            extNo_last = extNo;
            if (!(extNo > 2 && nmax >= 1 && nmin >= 1)) {                              // a trend
                if (extNo > 2 && GTID == 0) CTL->err |= ERR_NOEXT;                     // (PyEMD raises IndexError)
                finished = true; break;
            }
            if (nmax > kmax || nmin > kmax) { if (GTID == 0) CTL->err |= ERR_CAP; finished = true; break; }

            // ---- envelopes of the current imf
            extrema_write(cur, TT, N, es, kx0, ky0, kx1, ky1);
            if (GTID == 0) {
                CTL->degenerate = 0;
                CTL->kb[0] = 0; CTL->kb[1] = 0; CTL->kn[0] = 0; CTL->kn[1] = 0;
                MIRROR_POINTS(cur, TT, N, nmax, nmin, kx0, ky0, kx1, ky1);
            }
            SYNC();
            EnvBuf E0, E1;
            E0.kx = kx0 + CTL->kb[0]; E0.ky = ky0 + CTL->kb[0]; E0.sv = sv0; E0.rec = rc0; E0.n = CTL->kn[0];
            E1.kx = kx1 + CTL->kb[1]; E1.ky = ky1 + CTL->kb[1]; E1.sv = sv1; E1.rec = rc1; E1.n = CTL->kn[1];
            E0.mode = env_mode(E0.n);
            E1.mode = env_mode(E1.n);
#if EXTREMA_MODE == 1
            check_knots(E0, E1, TT, N, KNOTS_STRICT != 0);
#if KNOTS_STRICT == 0
            if (E0.mode >= 0 && E1.mode >= 0) {             // interp1d: stable sort of the knots; spline kinds reject duplicates
                sort_knots(E0, E1);
#if SPLINE_KIND == 2 || SPLINE_KIND == 3
                check_dups(E0, E1);
#endif
            }
#endif
#endif
            const bool bad_mode = (E0.mode < 0) || (E1.mode < 0);
            if (!bad_mode && CTL->degenerate == 0) spline_setup(E0, E1, vs);
            if (bad_mode || CTL->degenerate != 0) {         // PyEMD would raise (too few / non-increasing knots)
                if (GTID == 0) CTL->err |= ERR_SPLINE;
                finished = true;
                break;
            }
            const int bad_env = CTL->bad;                   // check_imf precondition on the pre-sift knots

            // ---- sift: new = old - 0.5*(max_env + min_env)
            sift_update(cur, nxt, TT, N, E0, E1);
            real_t* oldb = cur; real_t* newb = nxt; cur = newb; nxt = oldb;

            if (fixe != 0) {                                // PyEMD: fixed number of sifts, no re-count
                extNo_last = extNo;
                if (n >= fixe) { DBG(n, extNo, nzm); break; }
            } else if (fixe_h != 0) {
                es = extrema_count(cur, TT, N);
                const int e2 = es.tmax + es.tmin, z2 = es.tz;
                extNo_last = e2;
                have_counts = true;
                if (n == 1) continue;
                n_h = (abs(e2 - z2) < 2) ? (n_h + 1) : 0;
                if (n_h >= fixe_h) { DBG(n, e2, z2); break; }
            } else {                                        // Huang convergence (check_imf) AND |extNo - nzm| < 2
                es = extrema_count(cur, TT, N);
                const int e2 = es.tmax + es.tmin, z2 = es.tz;
                extNo_last = e2;
                have_counts = true;
                const bool f2 = (abs(e2 - z2) < 2);
                bool f1 = false;
                if (f2 && bad_env == 0) {                   // check_imf, with numpy's own sums (computed only now)
                    real_t st[3];
                    pw_sum<3>(pws, [&](int i, real_t* t) {
                        const real_t nw = newb[i], od = oldb[i];
                        const real_t d = nw - od;
                        t[0] = nw * nw; t[1] = d * d; t[2] = od * od;
                    }, st);
                    if (!(st[0] < 1.0e-10)) {
                        real_t omax, omin;
                        py_maxmin(oldb, N, omax, omin);
                        if (st[1] / (omax - omin) < svar_thr) f1 = true;                 // scaled variance
                        else if (st[1] / st[2] < energy_ratio_thr) f1 = true;            // energy ratio
                        else {                                                           // standard deviation
                            real_t sd;
                            pw_sum<1>(pws, [&](int i, real_t* t) {
                                const real_t nw = newb[i];
                                const real_t r = (nw - oldb[i]) / nw;
                                t[0] = r * r;
                            }, &sd);
                            f1 = (sd < std_thr);
                        }
                    }
                }
                if (f1 && f2) { DBG(n, e2, z2); break; }
            }
        }

        // ---- this imf is complete: store it, update the running sum (np.sum(IMF, axis=0): row after row), residue
        {
            // The loop can be left right after a pw_sum (check_imf) or at the MAX_ITERATION test that follows one.
            // pw_sum ends with each block evaluating the summation tree on its own, re-reading the tail terms of
            // oldb == nxt; with G > 1 a block must not overwrite nxt below before every block is done with that.
            SYNC();
            real_t* out_row = out_base + (size_t)imf_count * N;
            for (int j = GTID; j < N; j += GSIZE) {
                const real_t v = cur[j];
                out_row[j] = v;
                const real_t a = ((imf_count == 0) ? 0.0 : acc[j]) + v;      // numpy's reduction starts from 0.0
                acc[j] = a;
                nxt[j] = S[j] - a;
            }
            SYNC();
        }
        ++imf_count;
        { real_t* t = cur; cur = nxt; nxt = t; }            // cur = residue = S - sum(IMFs)
        // end_condition: np.max(tmp) - np.min(tmp) < range_thr  or  np.sum(np.abs(tmp)) < total_power_thr
        real_t gmax, gmin, gabs;
        np_maxmin(cur, N, gmax, gmin);
        pw_sum<1>(pws, [&](int i, real_t* t) { t[0] = fabs(cur[i]); }, &gabs);
        if ((gmax - gmin) < range_thr || gabs < total_power_thr || imf_count == max_imf) finished = true;
    }

    const bool exhausted = !finished;
    if (extNo_last <= 2 && imf_count > 0) imf_count -= 1;   // the last "imf" was a trend
    // final residue S - np.sum(imfs, axis=0); np.allclose(residue, 0) <=> no sample with |r| > 1e-8 (or NaN)
    real_t* res_row = out_base + (size_t)imf_count * N;
    long long notclose = 0;
    for (int j = GTID; j < N; j += GSIZE) {
        real_t a = 0.0;                                     // np.sum(imfs, axis=0): 0.0 + row 0 + row 1 + ...
        for (int k = 0; k < imf_count; ++k) a = a + out_base[(size_t)k * N + j];
        const real_t r = S[j] - a;
        res_row[j] = r;
        if (!(fabs(r) <= 1.0e-8)) notclose += 1;
    }
    GRed rr = gred_id(); rr.s = notclose;
    rr = group_reduce(rr);
    if (GTID == 0) {
        n_imfs_out[b] = (rr.s == 0) ? imf_count : imf_count + 1;
        exhausted_out[b] = exhausted ? 1 : 0;
        imf_count_out[b] = imf_count;
        err_out[b] = CTL->err;
        maxit_out[b] = n_maxit;
    }
}


#ifdef WITH_ENVELOPE_KERNEL   // compiled only for selftest() (it would add ~25% to the first-use compile time)
// ============================================================================================================
//  Diagnostic kernel: extrema + mirroring + spline envelopes of ONE sift step, no sifting (EMDBatchLauncher
//  .run_envelope_only).  One block per signal.  degenerate_out: 0 ok | 1 trend (extNo <= 2) | 3 PyEMD would raise
// ============================================================================================================
__device__ void eval_envelope(const EnvBuf& E, int e, const real_t* __restrict__ TT, real_t* __restrict__ out, int N)
{
    const int T = GSIZE;
    const int ms = (N + T - 1) / T;
#if USE_START_TABLE
    const int nsc = (N + ms - 1) / ms;
    start_table(E, e, ms, nsc);
    SYNC();
#endif
    const int gt = GTID;
    const int lo = gt * ms, hi = min(lo + ms, N);
    const int L = piece_last(E);
    if (lo < N) {
#if USE_START_TABLE
        int i = BSH.siv[(size_t)e * BSH.sivs + gt];
#else
        int i = find_piece(E, XPOS(TT, lo));
#endif
        i = (i < 0) ? 0 : ((i > L) ? L : i);
        Piece p; piece_load(E, i, p);
        for (int j = lo; j < hi; ++j) {
#if INTPOS
            while (i < L && j >= p.xri) { ++i; piece_load(E, i, p); }
            out[j] = piece_eval_i(E.mode, p, j);
#else
            const real_t x = XPOS(TT, j);
            while (i < L && x >= p.xr) { ++i; piece_load(E, i, p); }
            out[j] = piece_eval(E.mode, p, x);
#endif
        }
    }
    SYNC();
}

extern "C" __global__ __launch_bounds__(THREADS) void envelope_batch_kernel(
    const real_t* __restrict__ signals, const real_t* __restrict__ TT, int N,
    real_t* __restrict__ max_env_out, real_t* __restrict__ min_env_out,
    int* __restrict__ n_extrema_out, int* __restrict__ degenerate_out,
    real_t* __restrict__ max_knot_pos_out, real_t* __restrict__ max_knot_val_out,
    real_t* __restrict__ min_knot_pos_out, real_t* __restrict__ min_knot_val_out,
    int* __restrict__ n_mirrored_out,
    real_t* __restrict__ scratch, const long long* __restrict__ lay, int kcap, int* __restrict__ ctl_base, int ctl_words)
{
    group_setup(1, scratch, lay, ctl_base, ctl_words, 36 | (44 << 16), 16);
    const int b = blockIdx.x;
    const int kc = ALIGN4(kcap);
    const real_t* S = signals + (size_t)b * N;
    real_t* sc = scratch + (size_t)b * (size_t)lay[0];
    real_t* ek = sc + lay[1];
    real_t* kx0 = ek;                       real_t* ky0 = ek + kc;                 real_t* sv0 = ek + 2 * (size_t)kc;   real_t* rc0 = ek + 3 * (size_t)kc;
    real_t* kx1 = ek + ENV_KC * (size_t)kc; real_t* ky1 = kx1 + kc;                real_t* sv1 = kx1 + 2 * (size_t)kc;  real_t* rc1 = kx1 + 3 * (size_t)kc;
    real_t* vs = sc + lay[2];
    const ExtState es = extrema_count(S, TT, N);
    const int nmax = es.tmax, nmin = es.tmin;
    if (threadIdx.x == 0) { n_extrema_out[2 * b] = nmax; n_extrema_out[2 * b + 1] = nmin; }
    if (nmax + nmin <= 2 || nmax < 1 || nmin < 1) {
        if (threadIdx.x == 0) degenerate_out[b] = 1;
        return;
    }
    extrema_write(S, TT, N, es, kx0, ky0, kx1, ky1);
    if (threadIdx.x == 0) {
        CTL->degenerate = 0;
        CTL->kb[0] = 0; CTL->kb[1] = 0; CTL->kn[0] = 0; CTL->kn[1] = 0;
        MIRROR_POINTS(S, TT, N, nmax, nmin, kx0, ky0, kx1, ky1);
    }
    SYNC();
    EnvBuf E0, E1;
    E0.kx = kx0 + CTL->kb[0]; E0.ky = ky0 + CTL->kb[0]; E0.sv = sv0; E0.rec = rc0; E0.n = CTL->kn[0];
    E1.kx = kx1 + CTL->kb[1]; E1.ky = ky1 + CTL->kb[1]; E1.sv = sv1; E1.rec = rc1; E1.n = CTL->kn[1];
    E0.mode = env_mode(E0.n);
    E1.mode = env_mode(E1.n);
    for (int k = threadIdx.x; k < E0.n; k += blockDim.x) { max_knot_pos_out[(size_t)b * kcap + k] = E0.kx[k]; max_knot_val_out[(size_t)b * kcap + k] = E0.ky[k]; }
    for (int k = threadIdx.x; k < E1.n; k += blockDim.x) { min_knot_pos_out[(size_t)b * kcap + k] = E1.kx[k]; min_knot_val_out[(size_t)b * kcap + k] = E1.ky[k]; }
    if (threadIdx.x == 0) { n_mirrored_out[2 * b] = E0.n; n_mirrored_out[2 * b + 1] = E1.n; }
#if EXTREMA_MODE == 1
    check_knots(E0, E1, TT, N, KNOTS_STRICT != 0);
#if KNOTS_STRICT == 0
    if (E0.mode >= 0 && E1.mode >= 0) {
        sort_knots(E0, E1);
#if SPLINE_KIND == 2 || SPLINE_KIND == 3
        check_dups(E0, E1);
#endif
    }
#endif
#endif
    const bool bad_mode = (E0.mode < 0) || (E1.mode < 0);
    if (!bad_mode && CTL->degenerate == 0) spline_setup(E0, E1, vs);
    if (bad_mode || CTL->degenerate != 0) {
        if (threadIdx.x == 0) degenerate_out[b] = 3;
        return;
    }
    eval_envelope(E0, 0, TT, max_env_out + (size_t)b * N, N);
    eval_envelope(E1, 1, TT, min_env_out + (size_t)b * N, N);
    if (threadIdx.x == 0) degenerate_out[b] = 0;
}
#endif   // WITH_ENVELOPE_KERNEL
'''


# ----------------------------------------------------------------------------------------------------------------
# Ensemble helpers (EEMD / CEEMDAN): exact sequential sums over trials, and chunk-independent noise
# ----------------------------------------------------------------------------------------------------------------
_ACC_SOURCE = r'''
extern "C" __global__ void acc_rows(const double* __restrict__ imfs, const int* __restrict__ rowsel, int B, int slots,
                                    int N, double div, double* __restrict__ acc)
{
    // acc[j] += row_b[j] (or row_b[j] / div) for b = 0, 1, ..., B-1 IN THIS ORDER: numpy's axis-0 reduction order
    // (np.mean(rows, axis=0), E_IMF += IMFs, local_mean += imfs[-1] / trials), so the ensemble sums are numpy's.
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    double a = acc[j];
    for (int b = 0; b < B; ++b) {
        const int r = rowsel[b];
        if (r >= 0) {
            double v = imfs[((size_t)b * (size_t)slots + (size_t)r) * (size_t)N + (size_t)j];
            if (div > 0.0) v = v / div;
            a = a + v;
        }
    }
    acc[j] = a;
}
'''
_ACC_KERNEL: list = []


def accumulate_rows(acc: "cp.ndarray", imfs: "cp.ndarray", rowsel: "cp.ndarray", div: float = 0.0) -> None:
    """acc += imfs[b, rowsel[b]] (/ div) for every trial b with rowsel[b] >= 0, sequentially in trial order -- the
    order (and so the rounding) of numpy's axis-0 sums.  acc: [N] float64 (updated in place); imfs: [B, slots, N]."""
    if not _ACC_KERNEL:
        _ACC_KERNEL.append(cp.RawKernel(_ACC_SOURCE, "acc_rows", options=("--fmad=false",)))
    B, slots, N = imfs.shape
    if B == 0:
        return
    imfs = cp.ascontiguousarray(imfs)
    rowsel = cp.ascontiguousarray(cp.asarray(rowsel, dtype=cp.int32))
    threads = 256
    _ACC_KERNEL[0](((N + threads - 1) // threads,), (threads,),
                   (imfs, rowsel, np.int32(B), np.int32(slots), np.int32(N), np.float64(div), acc))


class NoiseStream:
    """Noise rows for trials 0, 1, 2, ..., requested in order, such that trial i's noise never depends on how the
    trials are chunked.  draw(rows) returns an array of shape (rows, N) (numpy or cupy).
    exact=True (numpy's RandomState): numpy's stream does not depend on how draws are split, so exactly the rows asked
    for are drawn -- the same numbers PyEMD draws, trial after trial.  exact=False (CuPy's generator, whose output does
    depend on the draw size): rows are drawn in fixed blocks of BLOCK, one generator call per block."""

    BLOCK = 64

    def __init__(self, draw, N: int, exact: bool = False):
        self._draw = draw
        self._N = int(N)
        self._exact = bool(exact)
        self._bi = -1
        self._block = None
        self._next = 0

    def rows(self, start: int, stop: int) -> "cp.ndarray":
        if self._exact:
            if start != self._next:
                raise RuntimeError("NoiseStream rows must be requested in order")
            self._next = stop
            return cp.asarray(self._draw(stop - start), dtype=cp.float64)
        out = cp.empty((stop - start, self._N), dtype=cp.float64)
        i = start
        while i < stop:
            bi = i // self.BLOCK
            if bi != self._bi:
                if bi != self._bi + 1:
                    raise RuntimeError("NoiseStream rows must be requested in order")
                self._block = cp.asarray(self._draw(self.BLOCK))
                self._bi = bi
            hi = min(stop, (bi + 1) * self.BLOCK)
            out[i - start:hi - start] = self._block[i - bi * self.BLOCK:hi - bi * self.BLOCK]
            i = hi
        return out


@dataclass
class EMDKernelConfig:
    """Compile-time constants of the kernel (a different value compiles, and caches, a different kernel).

    max_extrema        accepted for compatibility and IGNORED: there is no extrema-count limit.
    nbsym              boundary-mirroring depth (PyEMD's nbsym).
    max_iteration      default for PyEMD's MAX_ITERATION when EMDBatchLauncher.run() is called without one.
    spline_kind        PyEMD's spline_kind (case-insensitive).
    extrema_detection  PyEMD's extrema_detection: "simple" (default) or "parabol".
    max_imfs           initial IMF output slots per signal; the buffer grows whenever a signal needs more.
    threads_per_block  the largest block size the kernel is compiled for (a power of two, 32 .. 1024).
    """

    max_extrema: int = 1024
    nbsym: int = 2
    max_iteration: int = 1000
    max_imfs: int = 16
    threads_per_block: int = 256
    spline_kind: str = "cubic"
    extrema_detection: str = "simple"

    @property
    def dtype(self) -> "cp.dtype":
        return cp.dtype(cp.float64)

    @property
    def real_t_name(self) -> str:
        return "double"

    @property
    def itemsize(self) -> int:
        return 8

    # ------------------------------------------------------------------
    # memory model
    # ------------------------------------------------------------------
    @classmethod
    def for_device(cls, device=None, target_blocks_per_sm: int = 4, min_max_extrema: int = 1024, **fields):
        """Kept for API compatibility: returns EMDKernelConfig(**fields) (the launch geometry is chosen per call)."""
        return cls(**fields)

    def extrema_bound(self, N: int) -> int:
        """Upper bound on the number of maxima -- or of minima -- of an N-sample signal (two of a kind are never
        adjacent, in either extrema mode)."""
        return int(N) // 2 + 2

    def spill_real_arrays(self) -> int:
        return 0

    def spill_capacity(self, N: int) -> int:
        return 0

    def knot_capacity(self, N: int) -> int:
        """Slots of each envelope's knot arrays: all extrema of one kind plus the mirrored points on both sides."""
        return self.extrema_bound(N) + 4 * (self.nbsym + 2) + 8

    def ctl_words(self) -> int:
        """int32 words per signal of the zero-initialised control block: 4 barrier words + the group state struct."""
        F = self.nbsym + 2
        int_bytes = 4 * (16 + 4 * F)
        int_bytes = (int_bytes + 7) & ~7
        struct_bytes = int_bytes + 8 * (8 * F + 4)
        return _align4(4 + struct_bytes // 4 + 8)

    def layout(self, N: int, threads: int, G: int = 1) -> np.ndarray:
        """Scratch offsets (float64 slots) the kernel receives as `lay`:
        [signal stride, envelopes, solver verification, pairwise partials, start table, exchange slots,
         per-block tree nodes, per-block stride]."""
        N4 = _align4(N)
        kc = _align4(self.knot_capacity(N))
        rec_w = 5 if str(self.spline_kind).lower() == "quadratic" else 4
        env_kc = 3 + rec_w
        GS = int(G) * int(threads)
        L = int(pairwise_schedule(N)[0])
        off_env = 3 * N4
        off_vs = off_env + 2 * env_kc * kc
        off_P = off_vs + _align4(_VS_DOUBLES * GS)
        off_siv = off_P + 2 * L * 24
        off_gx = off_siv + _align4(GS)
        off_blk = off_gx + _align4(16 * G)
        bstride = _align4(2 * L * 3)
        sstride = off_blk + G * bstride
        return np.asarray([sstride, off_env, off_vs, off_P, off_siv, off_gx, off_blk, bstride], dtype=np.int64)

    def scratch_stride(self, N: int, threads: "int | None" = None, G: int = 1) -> int:
        threads = self.threads_per_block if threads is None else int(threads)
        return int(self.layout(N, threads, G)[0])

    def bytes_per_signal(self, N: int, slots: "int | None" = None, threads: "int | None" = None, G: int = 1) -> int:
        """Device memory one signal needs inside a launch with `slots` IMF slots: the IMF output, the working
        scratch, the control block and the small per-IMF diagnostic arrays."""
        slots = self.max_imfs if slots is None else int(slots)
        threads = self.threads_per_block if threads is None else int(threads)
        total = slots * int(N) * 8
        total += self.scratch_stride(N, threads, G) * 8
        total += self.ctl_words() * 4 + 3 * slots * 4 + 6 * 4
        return total

    def validate(self) -> None:
        t = self.threads_per_block
        if t < 32 or t > 1024 or (t & (t - 1)) != 0:
            raise ValueError(f"threads_per_block must be a power of two between 32 and 1024, got {t}")
        if self.nbsym < 1:
            raise ValueError(f"nbsym must be >= 1, got {self.nbsym}")
        if str(self.spline_kind).lower() not in SPLINE_KINDS:
            raise ValueError(
                f"No such interpolation method! spline_kind={self.spline_kind!r}; choose one of {sorted(SPLINE_KINDS)}"
            )
        if self.extrema_detection not in EXTREMA_MODES:
            raise ValueError(
                f"extrema_detection must be one of {sorted(EXTREMA_MODES)}, got {self.extrema_detection!r}"
            )
        if self.max_imfs < 1:
            raise ValueError(f"max_imfs must be >= 1, got {self.max_imfs}")

    def source(self) -> str:
        kind = str(self.spline_kind).lower()
        return (
            _CUDA_SOURCE.replace("@THREADS@", str(self.threads_per_block))
            .replace("@NBSYM@", str(self.nbsym))
            .replace("@SPLINE_KIND@", str(SPLINE_KINDS[kind]))
            .replace("@EXTREMA_MODE@", str(EXTREMA_MODES[self.extrema_detection]))
            .replace("@AKIMA_FORM@", str(_akima_form() if kind == "akima" else 1))
            .replace("@QUAD_FMA@", str(_quad_fma_form() if kind == "quadratic" else 1))
            .replace("@P3_FMA@", str(_p3_fma_form() if kind == "cubic" else 1))
        )

    def build(self) -> "EMDBatchLauncher":
        self.validate()
        return EMDBatchLauncher(self.source(), self)


_HELPER_PROBE: list = []


def _probe_int_helpers(source: str) -> None:
    """Run the kernel's integer helpers for doubles (zero test, 'safe' test, NaN test, order key) ON THE GPU against a
    set of edge-case values and compare every pairwise ordering with numpy.  These helpers replace FP64 comparisons
    in the extrema scan; a GPU compiler that mangles them would corrupt every decomposition, so this runs once per
    process, before the first launch, and raises instead of returning wrong IMFs."""
    if _HELPER_PROBE:
        return
    i, j = source.index("// @HELPERS_BEGIN"), source.index("// @HELPERS_END")
    head = "typedef double real_t;\n#define SAFE_EXP 560\n#define LL_MAX 0x7fffffffffffffffLL\n"
    code = head + source[i:j] + r'''
extern "C" __global__ void helper_probe(const double* __restrict__ x, int n, long long* __restrict__ key, int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long b = dbits(x[i]);
    key[i] = bnan(b) ? 0 : okey(b);
    flags[i] = (bz(b) ? 1 : 0) | (bsafe(b) ? 2 : 0) | (bnan(b) ? 4 : 0);
}
'''
    rng = np.random.RandomState(99)
    tiny = 2.0 ** (560 - 1023)
    special = [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, -2.2250738585072014e-308, tiny, -tiny, tiny * 0.5,
               -tiny * 0.5, np.nextafter(tiny, 0), 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 1e308, -1e308, np.inf, -np.inf, np.nan,
               1.7976931348623157e308, -1.7976931348623157e308]
    vals = np.concatenate([special, rng.randn(120) * np.exp(rng.randn(120) * 30), -np.abs(rng.randn(40)) * 1e-200])
    vals = vals.astype(np.float64)
    n = vals.size
    xd = cp.asarray(vals)
    keys = cp.zeros(n, dtype=cp.int64)
    flags = cp.zeros(n, dtype=cp.int32)
    k = cp.RawKernel(code, "helper_probe", options=("--std=c++11", "--fmad=false"))
    k(((n + 127) // 128,), (128,), (xd, np.int32(n), keys, flags))
    kh, fh = cp.asnumpy(keys), cp.asnumpy(flags)
    with np.errstate(invalid="ignore"):
        exp_flags = ((vals == 0) * 1 + (np.isfinite(vals) & (np.abs(vals) >= tiny)) * 2 + np.isnan(vals) * 4).astype(np.int32)
        ok = np.array_equal(fh, exp_flags)
        m = ~np.isnan(vals)
        v, kk = vals[m], kh[m]
        want = (v[:, None] > v[None, :]).astype(int) - (v[:, None] < v[None, :]).astype(int)   # numpy's ordering
        got = (kk[:, None] > kk[None, :]).astype(int) - (kk[:, None] < kk[None, :]).astype(int)   # the keys' ordering
        ok = ok and np.array_equal(want, got)
    if not ok:
        raise RuntimeError(
            "cuda_emd: this GPU's compiler mis-compiles the kernel's integer helpers for doubles (order key / zero / "
            "range tests); refusing to run rather than return wrong IMFs.  Please report your GPU, driver and CuPy "
            "versions.")
    _HELPER_PROBE.append(True)


def _env_int(name: str):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


class EMDBatchLauncher:
    """Compiles the kernel and launches it: G blocks per signal, one global scratch slice per signal."""


    def __init__(self, source: str, config: EMDKernelConfig):
        self.config = config
        # Fused multiply-adds must be off: every operation has to round like numpy's (the kernel uses explicit fma
        # only where the installed BLAS does).
        _probe_int_helpers(source)
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.kernel = self.module.get_function("emd_batch_kernel")
        self._source = source
        self._envelope_kernel = None
        self.last_slots = None
        self.last_imf_counts = None
        self.last_maxit = None
        self.last_geometry = None
        self._occ = {}
        self._tuned = {}
        self.last_tuning = {}
        self._occ_exact = False
        self._dev = None
        self._pw_dev = {}

    # ------------------------------------------------------------------
    # launch geometry
    # ------------------------------------------------------------------
    def block_threads(self, N: int) -> int:
        """Baseline block size for N-sample signals: about one thread per 16 samples, a whole number of warps,
        at most the compiled maximum."""
        cap = self.config.threads_per_block
        t = 32
        while t < cap and t * 16 < int(N):
            t *= 2
        return t

    def _device(self):
        if self._dev is None:
            try:
                dev = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                self._dev = (int(props["multiProcessorCount"]), int(props["maxThreadsPerMultiProcessor"]))
            except Exception:
                self._dev = (1, 2048)
        return self._dev

    def blocks_per_sm(self, threads: int) -> int:
        """Resident blocks per SM of the main kernel at this block size (CUDA occupancy API; a conservative estimate
        if that is unavailable)."""
        hit = self._occ.get(threads)
        if hit is not None:
            return hit
        occ = 0
        try:
            from cupy.cuda import driver
            fn = getattr(self.kernel, "kernel", self.kernel)
            occ = int(driver.occupancyMaxActiveBlocksPerMultiprocessor(fn.ptr, int(threads), 0))
        except Exception:
            occ = 0
        if occ > 0:
            self._occ_exact = True
        else:
            self._occ_exact = False
            try:
                regs = int(self.kernel.num_regs)
                per_block = ((regs * 32 + 255) // 256) * 256 * (threads // 32)
                occ = max(1, min(65536 // max(per_block, 1), self._device()[1] // threads))
            except Exception:
                occ = 1
        self._occ[threads] = occ
        return occ

    WARM_F, WARM_B = 36, 44              # solver warm-up rows (forward / backward); see cubic_solve
    TUNE_MAX_SAMPLES = 1 << 18           # auto-tune the launch shape only for launches up to this many samples

    def _warm(self) -> int:
        wf = _env_int("CUDA_EMD_WARM_F") or self.WARM_F
        wb = _env_int("CUDA_EMD_WARM_B") or self.WARM_B
        return (int(wf) & 0xffff) | ((int(wb) & 0xffff) << 16)

    def _capacity(self, T: int) -> int:
        return self._device()[0] * self.blocks_per_sm(T)

    def _groups_allowed(self) -> bool:
        return self._occ_exact and os.environ.get("CUDA_EMD_GROUPS", "1") != "0"

    def _heuristic(self, batch: int, N: int):
        """Launch shape without measurements: ~N/16 threads per block (more when the batch leaves the GPU idle); when
        the batch still leaves SMs idle, G blocks per signal with ~32 samples per thread."""
        sms, _ = self._device()
        cap_T = self.config.threads_per_block
        T = self.block_threads(N)
        while T < cap_T and 2 * T * 8 <= N:
            T2 = 2 * T
            if batch > sms * self.blocks_per_sm(T2):
                break
            if 2 * batch * T >= sms * self.blocks_per_sm(T) * T:
                break
            T = T2
        G = 1
        cap = self._capacity(T)
        if batch < cap and self._groups_allowed():
            G = max(1, min(cap // max(batch, 1), N // (T * 32), 64))
        return T, G

    def _shape_ok(self, batch: int, N: int, T: int, G: int) -> bool:
        """A launch shape the tuner may try: whole warps, at most the compiled block size, co-resident blocks only
        (group barrier safety), at least 2 samples per thread. Every valid shape gives bit-identical results."""
        if T < 32 or T > self.config.threads_per_block or G < 1 or G > 64:
            return False
        if G > 1 and not self._groups_allowed():
            return False
        if batch * G > self._capacity(T):
            return False
        return G == 1 and T <= max(32, N) or N >= 2 * T * G

    def _candidates(self, batch: int, N: int):
        """The heuristic shape's neighbourhood (kept for compatibility; the tuner walks it, see _autotuned_launch)."""
        T0, G0 = self._heuristic(batch, N)
        out = [(T0, G0)]
        for sh in self._neighbours(batch, N, T0, G0):
            out.append(sh)
        return out

    def _neighbours(self, batch: int, N: int, T: int, G: int):
        out = []
        for t, g in ((T, 2 * G), (T, G // 2), (2 * T, G), (T // 2, G), (2 * T, G // 2), (T // 2, 2 * G),
                     (T, 4 * G), (T, G // 4)):
            if (t, g) != (T, G) and (t, g) not in out and self._shape_ok(batch, N, t, g):
                out.append((t, g))
        return out

    def geometry(self, batch: int, N: int, G: "int | None" = None, threads: "int | None" = None):
        """(threads per block, blocks per signal, minimum solver chunk) for a [batch, N] launch.
        Results never depend on this choice; only the speed does."""
        env_g = _env_int("CUDA_EMD_BLOCKS_PER_SIGNAL")
        T, Gh = self._heuristic(batch, N)
        if threads:
            T = int(threads)
        if G is None:
            G = env_g
        if G is None:
            G = Gh if not threads else max(1, min(Gh, self._capacity(T) // max(batch, 1)))
        G = max(1, int(G))
        cap = self._capacity(T)
        if batch * G > cap:              # never launch more blocks than the GPU holds at once (group barrier safety)
            G = max(1, cap // max(batch, 1))
        busy = batch * G >= cap
        min_chunk = 64 if busy else 8    # throughput: long solver chunks (less warm-up work); latency: short chains
        return T, G, min_chunk

    def _pw(self, N: int):
        d = self._pw_dev.get(N)
        if d is None:
            d = cp.asarray(pairwise_schedule(N))
            self._pw_dev[N] = d
        return d

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(
        self,
        signals: "cp.ndarray",
        T: "cp.ndarray",
        max_imfs: "int | None" = None,
        allow_growth: bool = True,
        *,
        std_thr: float = 0.2,
        svar_thr: float = 0.001,
        energy_ratio_thr: float = 0.2,
        total_power_thr: float = 0.005,
        range_thr: float = 0.001,
        max_iteration: "int | None" = None,
        FIXE: int = 0,
        FIXE_H: int = 0,
        max_imf: int = -1,
        blocks_per_signal: "int | None" = None,
        threads: "int | None" = None,
    ) -> tuple:
        """Decompose a [batch, N] float64 array. The keyword-only arguments are PyEMD's EMD kwargs, with PyEMD's
        defaults; `max_imf` is the `max_imf` argument of PyEMD's emd().
        Returns (imfs, n_imfs, errors, debug_n, debug_extno, debug_nzm); imfs is [batch, slots, N].
        A signal that needs more IMF slots than the buffer has is re-run (alone) with a larger buffer, so the result is
        never truncated (allow_growth=False keeps the buffer size and leaves such signals unfinished, ERR_NONTERM)."""
        batch, N = signals.shape
        cfg = self.config
        if signals.dtype != cfg.dtype or T.dtype != cfg.dtype:
            raise ValueError(
                f"This launcher was compiled for real_t={cfg.real_t_name} (dtype {cfg.dtype}), "
                f"but got signals.dtype={signals.dtype}, T.dtype={T.dtype}."
            )
        slots = int(cfg.max_imfs if max_imfs is None else max_imfs)
        if slots < 1:
            raise ValueError(f"max_imfs must be >= 1, got {slots}")
        if max_imf is not None and int(max_imf) > 0:
            slots = min(slots, int(max_imf) + 1)
        params = dict(std_thr=std_thr, svar_thr=svar_thr, energy_ratio_thr=energy_ratio_thr,
                      total_power_thr=total_power_thr, range_thr=range_thr,
                      max_iteration=cfg.max_iteration if max_iteration is None else max_iteration,
                      FIXE=FIXE, FIXE_H=FIXE_H, max_imf=max_imf)
        outs = None
        if blocks_per_signal is None and threads is None and _env_int("CUDA_EMD_BLOCKS_PER_SIGNAL") is None:
            outs = self._autotuned_launch(signals, T, slots, params)
        if outs is None:
            outs = self._launch(signals, T, slots, params, blocks_per_signal, threads)
        # outs: imfs, n_imfs, err, dbg_n, dbg_e, dbg_z, exhausted, imf_count, maxit
        while True:
            ex = outs[6]
            if not bool(ex.any()):
                break
            idx = cp.nonzero(ex)[0]
            grown = self._grown_slots(slots, N, int(idx.size)) if allow_growth else slots
            if grown <= slots:
                outs[2][idx] |= ERR_NONTERM
                break
            sub = self._launch(cp.ascontiguousarray(signals[idx]), T, grown, params, blocks_per_signal, threads)
            outs = self._merge(outs, sub, idx, slots, grown)
            slots = grown
        self.last_slots = slots
        self.last_imf_counts = outs[7]
        self.last_maxit = outs[8]
        return outs[:6]

    TUNE_MAX_RUNS = 14

    def _autotuned_launch(self, signals, T, slots, params):
        """For a launch that leaves the GPU partly idle (small batch), find the fastest launch shape (threads per
        block x blocks per signal) by timing it on the real data, once per (batch, N): a greedy walk from the
        heuristic shape through its neighbours (x2 / /2 / x4 / /4 steps) until no neighbour is faster, at most
        TUNE_MAX_RUNS timed launches.  Every shape returns bit-identical results, so whichever launch is fastest also
        IS the answer of this call (no extra run).  Like cuDNN's benchmark mode, free of numerical side effects; the
        optimum depends on the GPU (SM count, FP64 rate, barrier latency), which is why it is measured, not assumed."""
        batch, N = signals.shape
        if os.environ.get("CUDA_EMD_AUTOTUNE", "1") == "0" or batch * N > self.TUNE_MAX_SAMPLES:
            return None
        # one tuning per (batch, N) octave: EEMD / CEEMDAN chunks and similar sizes reuse the measured shape
        key = (1 << max(0, (batch - 1).bit_length()), 1 << max(0, (N - 1).bit_length()))
        best = self._tuned.get(key)
        if best is not None:
            if self._shape_ok(batch, N, best[0], best[1]):
                return self._launch(signals, T, slots, params, best[1], best[0])
            return None
        try:
            T0, G0 = self._heuristic(batch, N)
            if not self._neighbours(batch, N, T0, G0):
                self._tuned[key] = (T0, G0)
                return None
        except Exception:
            return None
        # every shape returns the same bits, so the answer is taken from an untimed first launch (which also absorbs
        # one-off costs: memory-pool growth, first use of the module) and the timed launches only rank the shapes
        outs = self._launch(signals, T, slots, params, G0, T0)
        timed = {}

        def run(th, g):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            o = self._launch(signals, T, slots, params, g, th)
            stop.record()
            stop.synchronize()
            timed[(th, g)] = cp.cuda.get_elapsed_time(start, stop)
            del o

        cur = (T0, G0)
        run(*cur)
        improved = True
        while improved and len(timed) < self.TUNE_MAX_RUNS:
            improved = False
            for sh in self._neighbours(batch, N, *cur):
                if sh in timed or len(timed) >= self.TUNE_MAX_RUNS:
                    continue
                run(*sh)
                if timed[sh] < 0.97 * timed[cur]:          # 3 % margin: do not chase timing noise
                    cur, improved = sh, True
                    break                                   # first-improvement walk from the new shape
        self._tuned[key] = cur
        self.last_geometry = cur
        self.last_tuning = dict(sorted(timed.items()))
        return outs

    @staticmethod
    def _merge(outs, sub, idx, slots, grown):
        batch, _, N = outs[0].shape
        imfs = cp.zeros((batch, grown, N), dtype=outs[0].dtype)
        imfs[:, :slots] = outs[0]
        imfs[idx] = sub[0]
        dbg = []
        for k in (3, 4, 5):
            d = cp.full((batch, grown), -1, dtype=cp.int32)
            d[:, :slots] = outs[k]
            d[idx] = sub[k]
            dbg.append(d)
        flat = []
        for k in (1, 2, 6, 7, 8):
            a = outs[k].copy()
            a[idx] = sub[k]
            flat.append(a)
        return (imfs, flat[0], flat[1], dbg[0], dbg[1], dbg[2], flat[2], flat[3], flat[4])

    def _grown_slots(self, slots: int, N: int, n_signals: int) -> int:
        """Next IMF-buffer size for the signals that ran out of slots: double it, up to N + 2 rows (a decomposition
        that is still not finished then is one PyEMD would not finish either) and within free device memory."""
        hard_cap = int(N) + 2
        grown = min(max(2 * slots, slots + 4), hard_cap)
        try:
            free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
            per = self.config.bytes_per_signal(N, grown) * max(1, n_signals)
            while grown > slots and per > 0.8 * free:
                grown = max(slots, (slots + grown) // 2) if grown - slots > 1 else slots
                per = self.config.bytes_per_signal(N, grown) * max(1, n_signals)
        except Exception:
            pass
        return grown

    def _launch(self, signals, T, slots, params, blocks_per_signal=None, threads=None) -> tuple:
        batch, N = signals.shape
        cfg = self.config
        th, G, min_chunk = self.geometry(batch, N, blocks_per_signal, threads)
        self.last_geometry = (th, G)
        kcap = cfg.knot_capacity(N)
        lay = cfg.layout(N, th, G)
        lay_dev = cp.asarray(lay)
        scratch = cp.empty((batch * int(lay[0]),), dtype=cp.float64)
        cw = cfg.ctl_words()
        ctl = cp.zeros((batch * cw,), dtype=cp.int32)
        pw = self._pw(N)

        imfs_out = cp.zeros((batch, slots, N), dtype=cp.float64)
        n_imfs_out = cp.zeros((batch,), dtype=cp.int32)
        err_out = cp.zeros((batch,), dtype=cp.int32)
        exhausted_out = cp.zeros((batch,), dtype=cp.int32)
        imf_count_out = cp.zeros((batch,), dtype=cp.int32)
        maxit_out = cp.zeros((batch,), dtype=cp.int32)
        debug_n_out = cp.full((batch, slots), -1, dtype=cp.int32)
        debug_extno_out = cp.full((batch, slots), -1, dtype=cp.int32)
        debug_nzm_out = cp.full((batch, slots), -1, dtype=cp.int32)
        p = params
        self.kernel(
            (batch * G,), (th,),
            (signals, T, np.int32(N), np.int32(slots), np.int32(G),
             np.float64(p["std_thr"]), np.float64(p["svar_thr"]), np.float64(p["energy_ratio_thr"]),
             np.float64(p["total_power_thr"]), np.float64(p["range_thr"]),
             _i32(p["max_iteration"]), _i32(p["FIXE"]), _i32(p["FIXE_H"]), _i32(p["max_imf"]),
             np.int32(self._warm()), np.int32(min_chunk),
             imfs_out, n_imfs_out, err_out, exhausted_out, imf_count_out, maxit_out,
             scratch, lay_dev, np.int32(kcap), pw, ctl, np.int32(cw),
             debug_n_out, debug_extno_out, debug_nzm_out),
        )
        return (imfs_out, n_imfs_out, err_out, debug_n_out, debug_extno_out, debug_nzm_out, exhausted_out,
                imf_count_out, maxit_out)

    @property
    def envelope_kernel(self):
        """The diagnostic envelope kernel, compiled on first use (selftest) in a module of its own."""
        if self._envelope_kernel is None:
            mod = cp.RawModule(code="#define WITH_ENVELOPE_KERNEL 1\n" + self._source,
                               options=("--std=c++11", "--fmad=false"))
            self._envelope_kernel = mod.get_function("envelope_batch_kernel")
        return self._envelope_kernel

    def run_envelope_only(self, signals: "cp.ndarray", T: "cp.ndarray") -> tuple:
        """DIAGNOSTIC ONLY -- runs envelope_batch_kernel (extrema + mirror + spline, no sift loop) so it can be
        checked in isolation against PyEMD's extract_max_min_spline. Returns (max_env, min_env, n_extrema,
        degenerate, max_knot_pos, max_knot_val, min_knot_pos, min_knot_val, n_mirrored); `degenerate` is 0 ok,
        1 trend (extNo <= 2), 3 PyEMD would raise (too few / non-increasing knots)."""
        batch, N = signals.shape
        cfg = self.config
        if signals.dtype != cfg.dtype or T.dtype != cfg.dtype:
            raise ValueError(
                f"This launcher was compiled for real_t={cfg.real_t_name} (dtype {cfg.dtype}), "
                f"but got signals.dtype={signals.dtype}, T.dtype={T.dtype}."
            )
        th = self.block_threads(N)
        kcap = cfg.knot_capacity(N)
        lay = cfg.layout(N, th, 1)
        scratch = cp.empty((batch * int(lay[0]),), dtype=cp.float64)
        cw = cfg.ctl_words()
        ctl = cp.zeros((batch * cw,), dtype=cp.int32)
        max_env_out = cp.zeros((batch, N), dtype=cp.float64)
        min_env_out = cp.zeros((batch, N), dtype=cp.float64)
        n_extrema_out = cp.zeros((batch, 2), dtype=cp.int32)
        degenerate_out = cp.zeros((batch,), dtype=cp.int32)
        max_knot_pos_out = cp.zeros((batch, kcap), dtype=cp.float64)
        max_knot_val_out = cp.zeros((batch, kcap), dtype=cp.float64)
        min_knot_pos_out = cp.zeros((batch, kcap), dtype=cp.float64)
        min_knot_val_out = cp.zeros((batch, kcap), dtype=cp.float64)
        n_mirrored_out = cp.zeros((batch, 2), dtype=cp.int32)
        self.envelope_kernel(
            (batch,), (th,),
            (signals, T, np.int32(N), max_env_out, min_env_out, n_extrema_out, degenerate_out,
             max_knot_pos_out, max_knot_val_out, min_knot_pos_out, min_knot_val_out, n_mirrored_out,
             scratch, cp.asarray(lay), np.int32(kcap), ctl, np.int32(cw)),
        )
        return (max_env_out, min_env_out, n_extrema_out, degenerate_out,
                max_knot_pos_out, max_knot_val_out, min_knot_pos_out, min_knot_val_out,
                n_mirrored_out)

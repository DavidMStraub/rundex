#!/usr/bin/env python3
"""Benchmark all rundex public-API functions.

Three implementations are compared at a fixed loop order (default: 4):

  CRunDec  – C++ via the python-rundec SWIG wrapper
  rundex   – JAX eager (no JIT)
  JIT      – rundex after JAX compilation

Output columns
--------------
  CRunDec      mean call time (µs), or N/A if the CRunDec call is not
               straightforward (complex decpar setup)
  rundex       mean eager-JAX call time (µs)
  JIT          mean JIT-compiled call time (µs, after warmup)
  JIT compile  wall time of the very first JIT call (includes compilation, ms)
  C++/JIT      speedup of the C++ implementation relative to JIT (×)
  eager/JIT    speedup of eager JAX relative to JIT (×)

Usage
-----
    python benchmarks/bench.py [--reps N] [--loops L]
    (defaults: reps=2000, loops=4)

Notes
-----
* All rundex functions are JIT-traced on a single floating-point argument;
  other parameters are treated as static compile-time constants.
* JAX outputs are synchronised with block_until_ready() so async dispatch
  does not artificially deflate timings.
* python-rundec is loaded from the sibling directory ../rundec-python. Set
  the environment variable RUNDEC_PY to override the path.
* AlL2AlH, AlH2AlL, mL2mH, mH2mL use a CRunDec(6) instance with a freshly
  built TriplenfMmuArray each call (CRunDec resets the array in-place at the
  end of every call, so it cannot be shared across repetitions).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ── rundex ────────────────────────────────────────────────────────────────────
from rundex import (
    AlH2AlL,
    AlL2AlH,
    AlphasExact,
    AlphasLam,
    DecMqDownOS,
    DecMqUpOS,
    Decoupling,
    LamExpl,
    LamImpl,
    mH2mL,
    mL2mH,
    mMS2m1S,
    mMS2mMS,
    mMS2mOS,
    mOS2mMS,
    mOS2mSI,
)

# ── python-rundec (C++ wrapper, optional) ─────────────────────────────────────
_default_rundec_py = Path(__file__).resolve().parent.parent.parent / "rundec-python"
_RUNDEC_PY = Path(os.environ.get("RUNDEC_PY", _default_rundec_py))
sys.path.insert(0, str(_RUNDEC_PY))

_c5: object = None
_c6: object = None
HAS_CRUNDEC = False
try:
    import rundec as _crd  # type: ignore[import]

    _c5 = _crd.CRunDec(5)
    _c6 = _crd.CRunDec(6)
    HAS_CRUNDEC = True
except Exception as _exc:
    print(f"[warn] python-rundec not available: {_exc}", file=sys.stderr)


def _make_decpar(entries: list) -> "_crd.TriplenfMmuArray":  # type: ignore[name-defined]
    """Build a TriplenfMmuArray(4) from a list of Decoupling objects.

    CRunDec's threshold functions expect a fixed-size C array of length 4,
    with unused slots zeroed out (nf=0).  The function also *mutates* the
    array at the end (resets every slot to 0), so this helper must be called
    fresh for every timing repetition.
    """
    arr = _crd.TriplenfMmuArray(4)
    for j in range(4):
        slot = _crd.TriplenfMmu()
        slot.nf   = 0
        slot.Mth  = 0.0
        slot.muth = 0.0
        arr[j] = slot
    for j, dec in enumerate(entries):
        slot = _crd.TriplenfMmu()
        slot.nf   = dec.nf
        slot.Mth  = dec.Mth
        slot.muth = dec.muth
        arr[j] = slot
    return arr

# ── physical benchmark parameters ─────────────────────────────────────────────
_AS_MZ = jnp.float64(0.118)   # αs(MZ)
_MZ    = 91.18                 # Z-boson mass [GeV]
_MB    = jnp.float64(4.18)    # bottom MS-bar mass [GeV]
_MB_OS = jnp.float64(4.78)    # bottom on-shell mass [GeV]
_AS_MB = jnp.float64(0.2160)  # αs(MB_OS)
_MU_B  = 4.18                  # bottom mass scale [GeV]
_MU_T  = 800.0                 # high scale [GeV]
_AS_T  = jnp.float64(0.09)    # αs(MU_T)
_LAM   = jnp.float64(0.2086)  # ΛMS [GeV]
_NF    = 5

_DECPAR = [
    Decoupling(nf=5, Mth=4.8, muth=4.8),
    Decoupling(nf=6, Mth=173.0, muth=173.0),
]

# ── timing helpers ────────────────────────────────────────────────────────────

def _sync(r: object) -> None:
    if hasattr(r, "block_until_ready"):
        r.block_until_ready()


def _timeit(fn: Callable, max_reps: int, budget_s: float) -> tuple[float, int]:
    """Return (mean_s, actual_reps).

    Runs at most *max_reps* repetitions but caps total measurement time at
    *budget_s* seconds.  This prevents very slow functions (e.g. LamImpl
    in eager mode) from dominating the benchmark.
    """
    # Warmup
    _sync(fn())
    # Single-call estimate to choose reps
    t0 = time.perf_counter()
    _sync(fn())
    per_call = time.perf_counter() - t0
    reps = max(1, min(max_reps, int(budget_s / max(per_call, 1e-9))))
    t1 = time.perf_counter()
    for _ in range(reps):
        _sync(fn())
    return (time.perf_counter() - t1) / reps, reps


def _jit_bench(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    max_reps: int,
    budget_s: float,
) -> tuple[float, float, int]:
    """JIT *fn* on one JAX scalar; return (compile_s, mean_s, actual_reps).

    compile_s   – wall time of the very first call (XLA compilation included).
    mean_s      – mean of subsequent calls (compiled cache hit).
    actual_reps – how many post-compile calls were timed.
    """
    jitted = jax.jit(fn)
    t0 = time.perf_counter()
    jitted(x0).block_until_ready()
    compile_s = time.perf_counter() - t0
    # After warmup the function is fast; use full max_reps with budget guard
    t1 = time.perf_counter()
    jitted(x0).block_until_ready()
    per_call = time.perf_counter() - t1
    reps = max(1, min(max_reps, int(budget_s / max(per_call, 1e-9))))
    t2 = time.perf_counter()
    for _ in range(reps):
        jitted(x0).block_until_ready()
    mean_s = (time.perf_counter() - t2) / reps
    return compile_s, mean_s, reps


# ── benchmark specification ───────────────────────────────────────────────────

@dataclass
class Spec:
    name: str
    # One JAX-array argument → JAX array; used for eager *and* JIT timing.
    rdx_fn: Callable[[jnp.ndarray], jnp.ndarray]
    rdx_x0: jnp.ndarray
    # No-arg callable → Python float (CRunDec).  None = not benchmarked.
    crd_fn: Optional[Callable[[], float]] = None


def _build_specs(loops: int) -> list[Spec]:
    c = _c5  # CRunDec(5) object, or None

    def _crd(fn: Callable) -> Optional[Callable]:
        """Wrap a CRunDec call; return None if CRunDec is unavailable."""
        if not HAS_CRUNDEC:
            return None
        def safe():
            try:
                return fn()
            except Exception:
                return float("nan")
        return safe

    return [
        Spec(
            "LamExpl",
            lambda x, L=loops: LamExpl(x, _MZ, _NF, L),
            _AS_MZ,
            _crd(lambda L=loops: c.LamExpl(float(_AS_MZ), _MZ, L)),
        ),
        Spec(
            "LamImpl",
            lambda x, L=loops: LamImpl(x, _MZ, _NF, L),
            _AS_MZ,
            _crd(lambda L=loops: c.LamImpl(float(_AS_MZ), _MZ, L)),
        ),
        Spec(
            "AlphasLam",
            lambda x, L=loops: AlphasLam(x, _MZ, _NF, L),
            _LAM,
            _crd(lambda L=loops: c.AlphasLam(float(_LAM), _MZ, L)),
        ),
        Spec(
            "AlphasExact",
            lambda x, L=loops: AlphasExact(x, _MZ, _MU_B, _NF, L),
            _AS_MZ,
            _crd(lambda L=loops: c.AlphasExact(float(_AS_MZ), _MZ, _MU_B, L)),
        ),
        Spec(
            "AlL2AlH",
            lambda x, L=loops: AlL2AlH(x, _MU_B, _DECPAR, _MU_T, L),
            jnp.float64(0.22),
            # Decpar must be rebuilt each call because CRunDec resets it to 0.
            _crd(lambda L=loops: _c6.AlL2AlH(
                0.22, _MU_B, _make_decpar(_DECPAR), _MU_T, L)),
        ),
        Spec(
            "AlH2AlL",
            lambda x, L=loops: AlH2AlL(x, _MU_T, _DECPAR, _MU_B, L),
            _AS_T,
            _crd(lambda L=loops: _c6.AlH2AlL(
                float(_AS_T), _MU_T, _make_decpar(_DECPAR), _MU_B, L)),
        ),
        Spec(
            "DecMqUpOS",
            lambda x, L=loops: DecMqUpOS(x, 0.22, 173.0, 173.0, _NF, L),
            _MB,
            _crd(lambda L=loops: c.DecMqUpOS(float(_MB), 0.22, 173.0, 173.0, L)),
        ),
        Spec(
            "DecMqDownOS",
            lambda x, L=loops: DecMqDownOS(x, 0.22, 173.0, 173.0, _NF, L),
            _MB,
            _crd(lambda L=loops: c.DecMqDownOS(float(_MB), 0.22, 173.0, 173.0, L)),
        ),
        Spec(
            "mMS2mMS",
            lambda x, L=loops: mMS2mMS(x, 0.118, 0.224, _NF, L),
            _MB,
            _crd(lambda L=loops: c.mMS2mMS(float(_MB), 0.118, 0.224, L)),
        ),
        Spec(
            "mMS2mOS",
            lambda x, L=loops: mMS2mOS(x, mq=None, alphas=0.223, mu=_MU_B, nf=_NF, loops=L),
            _MB,
            _crd(lambda L=loops: c.mMS2mOS(float(_MB), None, 0.223, _MU_B, L)),
        ),
        Spec(
            "mOS2mMS",
            lambda x, L=loops: mOS2mMS(x, mq=None, alphas=0.223, mu=_MU_B, nf=_NF, loops=L),
            _MB_OS,
            _crd(lambda L=loops: c.mOS2mMS(float(_MB_OS), None, 0.223, _MU_B, L)),
        ),
        Spec(
            "mOS2mSI",
            lambda x, L=loops: mOS2mSI(x, mq=None, asM=_AS_MB, nf=_NF, loops=L),
            _MB_OS,
            _crd(lambda L=loops: c.mOS2mSI(float(_MB_OS), None, float(_AS_MB), L)),
        ),
        Spec(
            "mMS2m1S",
            lambda x, L=loops: mMS2m1S(x, mq=None, alphas=0.223, mu=_MU_B, nl=_NF - 1, loops=L),
            _MB,
            _crd(lambda L=loops: c.mMS2m1S(float(_MB), None, 0.223, _MU_B, _NF - 1, L)),
        ),
        Spec(
            "mL2mH",
            lambda x, L=loops: mL2mH(x, 0.22, _MU_B, _DECPAR, _MU_T, L),
            _MB,
            _crd(lambda L=loops: _c6.mL2mH(
                float(_MB), 0.22, _MU_B, _make_decpar(_DECPAR), _MU_T, L)),
        ),
        Spec(
            "mH2mL",
            lambda x, L=loops: mH2mL(x, _AS_T, _MU_T, _DECPAR, _MU_B, L),
            jnp.float64(160.0),
            _crd(lambda L=loops: _c6.mH2mL(
                160.0, float(_AS_T), _MU_T, _make_decpar(_DECPAR), _MU_B, L)),
        ),
    ]


# ── reporting ─────────────────────────────────────────────────────────────────

def _fmt_time(s: float) -> str:
    """Auto-scale time to ns / µs / ms with fixed 8-char numeric field."""
    if s < 1e-6:
        return f"{s * 1e9:8.2f} ns"
    if s < 1e-3:
        return f"{s * 1e6:8.2f} µs"
    if s < 1.0:
        return f"{s * 1e3:8.2f} ms"
    return f"{s:8.3f}  s"


def _fmt_x(ratio: float) -> str:
    return f"{ratio:6.1f}×"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000,
                    help="maximum repetitions per benchmark (default: 2000)")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="maximum seconds per timing phase per function (default: 1.0)")
    ap.add_argument("--loops", type=int, default=4,
                    help="loop order for all functions (default: 4)")
    args = ap.parse_args()

    specs = _build_specs(args.loops)

    # Column widths  (total ≤ 80 chars for standard terminals)
    W_NAME = 12
    W_TIME = 11
    W_COMP = 13
    W_SPD  =  9

    header = (
        f"{'Function':<{W_NAME}}"
        f"{'CRunDec':>{W_TIME}}"
        f"{'rundex':>{W_TIME}}"
        f"{'JIT':>{W_TIME}}"
        f"{'JIT compile':>{W_COMP}}"
        f"{'C++/JIT':>{W_SPD}}"
        f"{'eager/JIT':>{W_SPD}}"
    )
    sep = "─" * len(header)

    impl_label = "CRunDec + " if HAS_CRUNDEC else ""
    print(
        f"\nBenchmark ({impl_label}rundex eager + JIT) — "
        f"max_reps={args.reps}, budget={args.budget}s/phase, "
        f"loops={args.loops}, nf={_NF}"
    )
    print(sep)
    print(header)
    print(sep)

    for sp in specs:
        # ── CRunDec ──────────────────────────────────────────────────────────
        crd_s: Optional[float] = None
        if sp.crd_fn is not None:
            try:
                crd_s, _ = _timeit(sp.crd_fn, args.reps, args.budget)
                crd_col = _fmt_time(crd_s)
            except Exception:
                crd_col = f"{'ERR':>{W_TIME}}"
        else:
            crd_col = f"{'N/A':>{W_TIME}}"

        # ── rundex eager ─────────────────────────────────────────────────────
        eager_fn = lambda s=sp: s.rdx_fn(s.rdx_x0)  # type: ignore[misc]
        rdx_s, rdx_reps = _timeit(eager_fn, args.reps, args.budget)
        rdx_col = _fmt_time(rdx_s)

        # ── JIT ──────────────────────────────────────────────────────────────
        compile_s, jit_s, jit_reps = _jit_bench(
            sp.rdx_fn, sp.rdx_x0, args.reps, args.budget
        )
        jit_col  = _fmt_time(jit_s)
        comp_col = _fmt_time(compile_s)

        # ── speedups ─────────────────────────────────────────────────────────
        crd_spd_col   = _fmt_x(crd_s / jit_s) if crd_s is not None else f"{'—':>{W_SPD}}"
        eager_spd_col = _fmt_x(rdx_s / jit_s)

        print(
            f"{sp.name:<{W_NAME}}"
            f"{crd_col:>{W_TIME}}"
            f"{rdx_col:>{W_TIME}}"
            f"{jit_col:>{W_TIME}}"
            f"{comp_col:>{W_COMP}}"
            f"{crd_spd_col:>{W_SPD}}"
            f"{eager_spd_col:>{W_SPD}}"
        )

    print(sep)
    print(
        "C++/JIT   = CRunDec time / JIT time   (>1 → JIT is faster than C++)\n"
        "eager/JIT = rundex eager / JIT time   (>1 → JIT speeds up rundex)"
    )
    print()


if __name__ == "__main__":
    main()

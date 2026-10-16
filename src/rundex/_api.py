"""Public API wrappers and JAX-compiled entry points.

This module re-exports numerical kernels and provides optional JIT helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import jax

from . import _alphas as _acore
from . import _masses as _mcore

Decoupling = _acore.Decoupling
Scalar = _acore.Scalar

# Re-export public kernels directly (no pass-through wrapper duplication).
LamExpl = _acore.LamExpl
LamImpl = _acore.LamImpl
AlphasLam = _acore.AlphasLam
AlphasExact = _acore.AlphasExact
AlL2AlH = _acore.AlL2AlH
AlH2AlL = _acore.AlH2AlL

mMS2mMS = _mcore.mMS2mMS
mMS2mOS = _mcore.mMS2mOS
mMS2m1S = _mcore.mMS2m1S
mOS2mMS = _mcore.mOS2mMS
mOS2mSI = _mcore.mOS2mSI
DecMqUpOS = _mcore.DecMqUpOS
DecMqDownOS = _mcore.DecMqDownOS
mL2mH = _mcore.mL2mH
mH2mL = _mcore.mH2mL


def _freeze_decpar(decpar: Iterable[object]) -> tuple[Decoupling, ...]:
    return tuple(_acore._normalize_decpar(decpar))


def _freeze_mq(mq: object) -> tuple[tuple[Scalar, Scalar], ...]:
    return tuple(_mcore._normalize_mq(mq))


# ---------------------------------------------------------------------------
# Cached compiled wrappers (static nf/loops boundary)
# ---------------------------------------------------------------------------


def compiled_LamExpl(nf: int, loops: int) -> Callable[[float, float], jax.Array]:
    """Return a compiled `LamExpl` with static `nf` and `loops`."""
    return jax.jit(partial(_acore.LamExpl, nf=nf, loops=loops))


def compiled_LamImpl(nf: int, loops: int) -> Callable[[float, float], jax.Array]:
    """Return a compiled `LamImpl` with static `nf` and `loops`."""
    return jax.jit(partial(_acore.LamImpl, nf=nf, loops=loops))


def compiled_AlphasLam(nf: int, loops: int) -> Callable[[float, float], jax.Array]:
    """Return a compiled `AlphasLam` with static `nf` and `loops`."""
    return jax.jit(partial(_acore.AlphasLam, nf=nf, loops=loops))


def compiled_AlphasExact(nf: int, loops: int) -> Callable[[Scalar, float, float], jax.Array]:
    """Return a compiled `AlphasExact` with static `nf` and `loops`."""
    return jax.jit(partial(_acore.AlphasExact, nf=nf, loops=loops))


def compiled_AlL2AlH(
    decpar: Iterable[object], loops: int
) -> Callable[[float, float, float], jax.Array]:
    """Return compiled `AlL2AlH` with static thresholds and loop order."""
    frozen = _freeze_decpar(decpar)
    return jax.jit(partial(_acore.AlL2AlH, decpar=frozen, loops=loops))


def compiled_AlH2AlL(
    decpar: Iterable[object], loops: int
) -> Callable[[float, float, float], jax.Array]:
    """Return compiled `AlH2AlL` with static thresholds and loop order."""
    frozen = _freeze_decpar(decpar)
    return jax.jit(partial(_acore.AlH2AlL, decpar=frozen, loops=loops))


def compiled_mMS2mMS(nf: int, loops: int) -> Callable[[Scalar, Scalar, Scalar], Scalar]:
    """Return a compiled `mMS2mMS` with static `nf` and `loops`."""
    return jax.jit(partial(_mcore.mMS2mMS, nf=nf, loops=loops))


def compiled_DecMqUpOS(nf: int, loops: int) -> Callable[[Scalar, Scalar, float, float], Scalar]:
    """Return a compiled `DecMqUpOS` with static `nf` and `loops`."""
    return jax.jit(partial(_mcore.DecMqUpOS, nf=nf, loops=loops))


def compiled_DecMqDownOS(nf: int, loops: int) -> Callable[[Scalar, Scalar, float, float], Scalar]:
    """Return a compiled `DecMqDownOS` with static `nf` and `loops`."""
    return jax.jit(partial(_mcore.DecMqDownOS, nf=nf, loops=loops))


def compiled_mL2mH(
    decpar: Iterable[object], loops: int
) -> Callable[[float, float, float, float], Scalar]:
    """Return compiled `mL2mH` with static thresholds and loop order."""
    frozen = _freeze_decpar(decpar)
    return jax.jit(partial(_mcore.mL2mH, decpar=frozen, loops=loops))


def compiled_mH2mL(
    decpar: Iterable[object], loops: int
) -> Callable[[float, float, float, float], Scalar]:
    """Return compiled `mH2mL` with static thresholds and loop order."""
    frozen = _freeze_decpar(decpar)
    return jax.jit(partial(_mcore.mH2mL, decpar=frozen, loops=loops))


def compiled_mOS2mMS(
    nf: int,
    loops: int,
    mq: object = None,
    fdelm: float = 1.0,
) -> Callable[[float, Scalar, float], Scalar]:
    """Return compiled `mOS2mMS` with static `nf`, `loops`, and light-quark data."""
    frozen_mq = _freeze_mq(mq)
    return jax.jit(partial(_mcore.mOS2mMS, mq=frozen_mq, nf=nf, loops=loops, fdelm=fdelm))


def compiled_mOS2mSI(
    nf: int,
    loops: int,
    mq: object = None,
    fdelm: float = 1.0,
) -> Callable[[float, Scalar], Scalar]:
    """Return compiled `mOS2mSI` with static `nf`, `loops`, and light-quark data."""
    frozen_mq = _freeze_mq(mq)
    return jax.jit(partial(_mcore.mOS2mSI, mq=frozen_mq, nf=nf, loops=loops, fdelm=fdelm))


def compiled_mMS2mOS(
    nf: int,
    loops: int,
    mq: object = None,
    fdelm: float = 1.0,
) -> Callable[[float, Scalar, float], Scalar]:
    """Return compiled `mMS2mOS` with static `nf`, `loops`, and light-quark data."""
    frozen_mq = _freeze_mq(mq)
    return jax.jit(partial(_mcore.mMS2mOS, mq=frozen_mq, nf=nf, loops=loops, fdelm=fdelm))


def compiled_mMS2m1S(
    nl: int,
    loops: int,
    mq: object = None,
    fdelm: float = 1.0,
) -> Callable[[float, Scalar, float], Scalar]:
    """Return compiled `mMS2m1S` with static `nl`, `loops`, and light-quark data."""
    frozen_mq = _freeze_mq(mq)
    return jax.jit(partial(_mcore.mMS2m1S, mq=frozen_mq, nl=nl, loops=loops, fdelm=fdelm))

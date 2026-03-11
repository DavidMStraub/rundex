"""
JAX jittability and gradability checks for public API functions.
"""

import jax
import pytest

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

DECPAR = [
    Decoupling(nf=6, Mth=173.0, muth=173.0),
    Decoupling(nf=5, Mth=4.8, muth=4.8),
]


def _jit_and_grad(fun, x0):
    jitted = jax.jit(fun)
    _ = jitted(x0)
    return jax.grad(fun)(x0)


@pytest.mark.xfail(reason="LamImpl uses lax.while_loop which does not support reverse-mode AD")
def test_LamImpl_jit_grad():
    f = lambda a: LamImpl(a, 91.18, 5, 4)
    _ = _jit_and_grad(f, 0.118)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_DecMqUpDownOS_jit_grad(loops):
    f_up = lambda mq: DecMqUpOS(mq, 0.22, 173.0, 173.0, 5, loops)
    f_down = lambda mq: DecMqDownOS(mq, 0.22, 173.0, 173.0, 5, loops)
    _ = _jit_and_grad(f_up, 4.18)
    _ = _jit_and_grad(f_down, 4.18)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_mMS2mMS_jit_grad(loops):
    f = lambda m: mMS2mMS(m, 0.118, 0.224, 5, loops)
    _ = _jit_and_grad(f, 4.18)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_LamExpl_jit(loops):
    f = lambda a: LamExpl(a, 91.18, 5, loops)
    _ = _jit_and_grad(f, 0.118)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_AlphasLam_jit(loops):
    f = lambda lam: AlphasLam(lam, 91.18, 5, loops)
    _ = _jit_and_grad(f, 0.2086)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_AlphasExact_jit(loops):
    f = lambda a: AlphasExact(a, 91.18, 10.0, 5, loops)
    _ = _jit_and_grad(f, 0.118)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_AlL2AlH_jit(loops):
    f = lambda a: AlL2AlH(a, 4.0, DECPAR, 800.0, loops)
    _ = _jit_and_grad(f, 0.22)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_AlH2AlL_jit(loops):
    f = lambda a: AlH2AlL(a, 800.0, DECPAR, 4.0, loops)
    _ = _jit_and_grad(f, 0.09)


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mMS2mOS_jit(loops):
    f = lambda m: mMS2mOS(m, mq=None, alphas=0.223, mu=4.18, nf=5, loops=loops)
    _ = _jit_and_grad(f, 4.18)


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mOS2mMS_jit(loops):
    f = lambda m: mOS2mMS(m, mq=None, alphas=0.223, mu=4.18, nf=5, loops=loops)
    _ = _jit_and_grad(f, 4.18)


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mOS2mSI_jit(loops):
    f = lambda m: mOS2mSI(m, mq=None, asM=0.216, nf=5, loops=loops)
    _ = _jit_and_grad(f, 4.78)


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mMS2m1S_jit(loops):
    f = lambda m: mMS2m1S(m, mq=None, alphas=0.223, mu=4.18, nl=5, loops=loops)
    _ = _jit_and_grad(f, 4.18)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_mL2mH_jit(loops):
    f = lambda m: mL2mH(m, 0.22, 4.0, DECPAR, 800.0, loops)
    _ = _jit_and_grad(f, 4.18)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_mH2mL_jit(loops):
    f = lambda m: mH2mL(m, 0.09, 800.0, DECPAR, 4.0, loops)
    _ = _jit_and_grad(f, 160.0)

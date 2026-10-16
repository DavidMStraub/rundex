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


# ---------------------------------------------------------------------------
# Traced scales and forward-mode AD
# ---------------------------------------------------------------------------


def _central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("loops", [1, 3, 5])
def test_AlphasExact_jacfwd_matches_jacrev(loops):
    f = lambda a, mu0, mu: AlphasExact(a, mu0, mu, 5, loops)
    args = (0.118, 91.18, 10.0)
    fwd = jax.jacfwd(f, argnums=(0, 1, 2))(*args)
    rev = jax.jacrev(f, argnums=(0, 1, 2))(*args)
    for d_fwd, d_rev in zip(fwd, rev, strict=True):
        assert d_fwd == pytest.approx(d_rev, rel=1e-12)


@pytest.mark.parametrize("loops", [1, 3, 5])
def test_AlphasExact_derivatives_match_finite_differences(loops):
    a, mu0, mu = 0.118, 91.18, 10.0
    grads = jax.grad(lambda *x: AlphasExact(*x, 5, loops), argnums=(0, 1, 2))(a, mu0, mu)
    fd = (
        _central_diff(lambda x: AlphasExact(x, mu0, mu, 5, loops), a),
        _central_diff(lambda x: AlphasExact(a, x, mu, 5, loops), mu0),
        _central_diff(lambda x: AlphasExact(a, mu0, x, 5, loops), mu),
    )
    for g, d in zip(grads, fd, strict=True):
        assert g == pytest.approx(d, rel=1e-6)


def test_AlphasExact_equal_scales_traced():
    # mu0 == mu must neither branch in Python nor lose the scale dependence.
    f = jax.jit(lambda a, mu0, mu: AlphasExact(a, mu0, mu, 5, 4))
    assert f(0.118, 91.18, 91.18) == pytest.approx(0.118, rel=1e-14)
    d_a, d_mu0, d_mu = jax.jacfwd(f, argnums=(0, 1, 2))(0.118, 91.18, 91.18)
    assert d_a == pytest.approx(1.0, rel=1e-14)
    assert d_mu == pytest.approx(-d_mu0, rel=1e-14)
    fd = _central_diff(lambda mu: AlphasExact(0.118, 91.18, mu, 5, 4), 91.18)
    assert d_mu == pytest.approx(fd, rel=1e-6)


def test_AlphasExact_second_derivative():
    f = lambda mu: AlphasExact(0.118, 91.18, mu, 5, 4)
    d2_fwd = jax.jacfwd(jax.jacfwd(f))(10.0)
    d2_rev = jax.grad(jax.grad(f))(10.0)
    assert d2_fwd == pytest.approx(d2_rev, rel=1e-10)
    assert d2_fwd == pytest.approx(_central_diff(jax.grad(f), 10.0), rel=1e-6)


@pytest.mark.parametrize("loops", [1, 3, 5])
def test_mL2mH_traced_threshold_scale(loops):
    # m_b(m_b) used both as the input mass and as the decoupling scale.
    def f(mb):
        decpar = [Decoupling(nf=6, Mth=173.0, muth=173.0), Decoupling(nf=5, Mth=4.8, muth=mb)]
        return mL2mH(mb, 0.22, mb, decpar, 800.0, loops)

    fwd = jax.jacfwd(f)(4.18)
    rev = jax.jit(jax.grad(f))(4.18)
    assert fwd == pytest.approx(rev, rel=1e-12)
    assert fwd == pytest.approx(_central_diff(f, 4.18), rel=1e-6)


def test_AlH2AlL_traced_decpar_tuple():
    f = lambda mth: AlH2AlL(0.09, 800.0, [(6, mth, mth), (5, 4.8, 4.8)], 4.0, 4)
    assert jax.jacfwd(f)(173.0) == pytest.approx(_central_diff(f, 173.0), rel=1e-6)


def test_mOS2mMS_traced_light_quark_mass():
    f = lambda mc: mOS2mMS(4.78, mq=[(mc, 3.0)], alphas=0.223, mu=4.18, nf=5, loops=3)
    assert jax.jacfwd(f)(1.28) == pytest.approx(_central_diff(f, 1.28), rel=1e-6)

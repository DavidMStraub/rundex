"""
Tests for decoupling and threshold running of αs.

Reference values are generated with rundec-python (CRunDec wrapper). We use the
same building blocks as the C++ implementation:
`AlphasExact` combined with `DecAsUpOS` / `DecAsDownOS` at thresholds.
"""

import pytest

from rundex import AlH2AlL, AlL2AlH, Decoupling

rundec = pytest.importorskip("rundec")


def rel_err(a, b):
    if b == 0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def _ref_all2alh(alphas_low: float, mu1: float, mu2: float, loops: int) -> float:
    c = rundec.CRunDec()
    thresholds = [(4, 1.5, 1.5), (5, 4.8, 4.2)]
    asini = alphas_low
    muini = mu1
    for nf, mth, muth in thresholds:
        as_at_muth = c.AlphasExact(asini, muini, muth, nf - 1, loops)
        asini = c.DecAsUpOS(as_at_muth, mth, muth, nf - 1, loops)
        muini = muth
    return c.AlphasExact(asini, muini, mu2, thresholds[-1][0], loops)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_all2alh_vs_crundec(loops):
    decpar = [
        Decoupling(nf=4, Mth=1.5, muth=1.5),
        Decoupling(nf=5, Mth=4.8, muth=4.2),
    ]
    result = AlL2AlH(0.35, 2.0, decpar, 100.0, loops)
    ref = _ref_all2alh(0.35, 2.0, 100.0, loops)
    assert rel_err(result, ref) < 1e-8, f"loops={loops}: got {result:.12e}, ref {ref:.12e}"


def _ref_alh2all(alphas_high: float, mu1: float, mu2: float, loops: int) -> float:
    c = rundec.CRunDec()
    thresholds = [(4, 1.5, 1.5)]
    asini = alphas_high
    muini = mu1
    for nf, mth, muth in thresholds:
        as_at_muth = c.AlphasExact(asini, muini, muth, nf, loops)
        asini = c.DecAsDownOS(as_at_muth, mth, muth, nf - 1, loops)
        muini = muth
    return c.AlphasExact(asini, muini, mu2, thresholds[-1][0] - 1, loops)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_alh2all_vs_crundec(loops):
    decpar = [Decoupling(nf=4, Mth=1.5, muth=1.5)]
    result = AlH2AlL(0.3, 5.0, decpar, 2.0, loops)
    ref = _ref_alh2all(0.3, 5.0, 2.0, loops)
    assert rel_err(result, ref) < 1e-8, f"loops={loops}: got {result:.12e}, ref {ref:.12e}"


def test_invalid_loops():
    decpar = [Decoupling(nf=4, Mth=1.5, muth=1.5)]
    with pytest.raises(ValueError):
        AlL2AlH(0.35, 2.0, decpar, 100.0, 0)
    with pytest.raises(ValueError):
        AlH2AlL(0.3, 5.0, decpar, 2.0, 6)

"""
Tests for mL2mH and mH2mL (mass running across thresholds).
"""

import pytest

from rundex import Decoupling, mH2mL, mL2mH

DECPAR = [
    Decoupling(nf=6, Mth=173.0, muth=173.0),
    Decoupling(nf=5, Mth=4.8, muth=4.8),
]

REF_ML2MH = {
    1: 2.635906122716044,
    2: 2.4771493060030503,
    3: 2.4572231979798187,
    4: 2.4538635812012575,
    5: 2.4530131504060306,
}

REF_MH2ML = {
    1: 249.7539297434719,
    2: 272.0791437008696,
    3: 274.59404104298693,
    4: 274.96800925360765,
    5: 274.9411346511245,
}


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_mL2mH_vs_crundec(loops):
    result = mL2mH(4.18, 0.22, 4.0, DECPAR, 800.0, loops)
    assert result == pytest.approx(REF_ML2MH[loops], rel=1e-8)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_mH2mL_vs_crundec(loops):
    result = mH2mL(160.0, 0.09, 800.0, DECPAR, 4.0, loops)
    assert result == pytest.approx(REF_MH2ML[loops], rel=1e-8)

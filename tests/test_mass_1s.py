"""
Tests for mMS2m1S (1S mass).
"""

import pytest

from rundex import mMS2m1S

ASMU = 0.223
MU = 4.18
NL = 5
MMS = 4.18

REF_MS2_1S = {
    0: 4.18,
    1: 4.529419787520922,
    2: 4.635059643220996,
    3: 4.65937619782682,
    4: 4.643950995375138,
}


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mMS2m1S_vs_crundec(loops):
    result = mMS2m1S(MMS, mq=None, alphas=ASMU, mu=MU, nl=NL, loops=loops, fdelm=1.0)
    assert result == pytest.approx(REF_MS2_1S[loops], rel=1e-8)

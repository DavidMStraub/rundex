"""
Tests for MS-bar mass running (mMS2mMS).
"""

import pytest

from rundex import AlphasExact, mMS2mMS

# Reference values from rundec-python 0.7 (CRunDec C++ wrapper)
REF_AS1 = {
    1: 0.21216479069014285,
    2: 0.22357627090070306,
    3: 0.22433389857477676,
    4: 0.22461041542615612,
    5: 0.2246228676826178,
}

REF_M1 = {
    1: 5.676895190704048,
    2: 6.054930692335902,
    3: 6.097682071000059,
    4: 6.102585215022403,
    5: 6.103016439151257,
}

AS0 = 0.118
MU0 = 91.18
MU1 = 4.18
NF = 5
M0 = 4.18


def test_identity_same_alphas():
    m0 = 4.18
    a0 = 0.225
    for loops in range(1, 6):
        assert mMS2mMS(m0, a0, a0, nf=5, loops=loops) == pytest.approx(m0)


def test_loops_zero_returns_input():
    assert mMS2mMS(4.18, 0.2, 0.25, nf=5, loops=0) == pytest.approx(4.18)


def test_invalid_loops():
    with pytest.raises(ValueError):
        mMS2mMS(4.18, 0.2, 0.25, nf=5, loops=-1)
    with pytest.raises(ValueError):
        mMS2mMS(4.18, 0.2, 0.25, nf=5, loops=6)


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 5])
def test_vs_crundec_mms2mms(loops):
    as1 = AlphasExact(AS0, MU0, MU1, NF, loops)
    assert as1 == pytest.approx(REF_AS1[loops], rel=1e-8)
    m1 = mMS2mMS(M0, AS0, as1, NF, loops)
    assert m1 == pytest.approx(REF_M1[loops], rel=1e-8)

"""
Tests for mOS2mSI: on-shell mass → self-invariant MS-bar mass m_MS(m_MS).

Reference values generated with CRunDec (C++ wrapper via rundec-python),
CRunDec::mOS2mSI(mOS, NULL, asM, nf=5, loops).
"""

import pytest

from rundex import mOS2mSI

# Physical parameters (bottom quark)
M_OS = 4.78  # on-shell mass [GeV]
ASM = 0.2160  # αs(M_OS)
NF = 5

# Reference values from CRunDec (nf=5, mq=None, fdelm=1)
REF = {
    0: 4.78,
    1: 4.341801878283946,
    2: 4.172326874983236,
    3: 4.049807523236252,
    4: 3.935271243842368,
}


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mOS2mSI_vs_crundec(loops):
    """Numerical match to CRunDec at all loop orders."""
    result = mOS2mSI(M_OS, mq=None, asM=ASM, nf=NF, loops=loops)
    assert result == pytest.approx(REF[loops], rel=1e-8)


def test_mOS2mSI_loops0_returns_input():
    """At 0 loops (no αs corrections) the result equals m_OS."""
    assert mOS2mSI(4.78, mq=None, asM=0.216, nf=5, loops=0) == pytest.approx(4.78)


def test_mOS2mSI_invalid_loops():
    with pytest.raises(ValueError):
        mOS2mSI(4.78, mq=None, asM=0.216, nf=5, loops=-1)
    with pytest.raises(ValueError):
        mOS2mSI(4.78, mq=None, asM=0.216, nf=5, loops=5)


def test_mOS2mSI_4loop_invalid_nf():
    """4-loop result is only available for nf=4..6."""
    with pytest.raises(ValueError):
        mOS2mSI(4.78, mq=None, asM=0.216, nf=3, loops=4)


def test_mOS2mSI_fdelm_zero():
    """fdelm=0 kills the 4-loop non-log term; result must differ from fdelm=1."""
    r1 = mOS2mSI(M_OS, mq=None, asM=ASM, nf=NF, loops=4, fdelm=1.0)
    r0 = mOS2mSI(M_OS, mq=None, asM=ASM, nf=NF, loops=4, fdelm=0.0)
    assert r1 != pytest.approx(r0)


def test_mOS2mSI_smaller_than_mOS():
    """MS mass at its own scale is always smaller than the OS mass."""
    for loops in [1, 2, 3, 4]:
        result = mOS2mSI(M_OS, mq=None, asM=ASM, nf=NF, loops=loops)
        assert result < M_OS

"""
Tests for OS ↔ MS mass conversions (mMS2mOS, mOS2mMS).
"""

import pytest

from rundex import mMS2mOS, mOS2mMS

ASMU = 0.223
MU = 4.18
NF = 5
MMS = 4.18

REF_MS2OS = {
    0: 4.18,
    1: 4.575612503076478,
    2: 4.7710182305616655,
    3: 4.912172383408958,
    4: 5.043300204820706,
}

REF_OS2MS = {
    0: 4.18,
    1: 3.784387496923521,
    2: 3.570260566975363,
    3: 3.4185861926316843,
    4: 3.282890917409237,
}


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mMS2mOS_vs_crundec(loops):
    result = mMS2mOS(MMS, mq=None, alphas=ASMU, mu=MU, nf=NF, loops=loops)
    assert result == pytest.approx(REF_MS2OS[loops], rel=1e-8)


@pytest.mark.parametrize("loops", [0, 1, 2, 3, 4])
def test_mOS2mMS_vs_crundec(loops):
    result = mOS2mMS(MMS, mq=None, alphas=ASMU, mu=MU, nf=NF, loops=loops)
    assert result == pytest.approx(REF_OS2MS[loops], rel=1e-8)

"""
Running of MS-bar quark masses.

Public functions
----------------
mMS2mMS(mmu0, alphas0, alphas1, nf, loops)
    Run the MS-bar mass from αs(μ0) to αs(μ1) using the closed-form
    solution of the RGE (Eqs. (9)–(10) in RunDec).

Conventions
-----------
* αs ≡ α_s (not α_s/π). Internally we use a = α_s/π.
* `loops` is a plain Python int (0..5).
"""

from __future__ import annotations

import math
from typing import Iterable

import jax.numpy as jnp
from jax.scipy.special import spence

from ._alphas import (
    AlphasExact,
    DecAsDownOS,
    DecAsUpOS,
    _check_consecutive_nfs,
    _normalize_decpar,
)
from ._coeffs import (
    beta0,
    beta1,
    beta2,
    beta3,
    beta4,
    gamma0,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
)

_PI = math.pi
_LOG2 = math.log(2.0)
_ZETA2 = _PI * _PI / 6.0
_ZETA3 = 1.2020569031595942854
_ZETA4 = _PI**4 / 90.0
_ZETA5 = 1.0369277551433699263
_ZETA6 = _PI**6 / 945.0

_CF = 4.0 / 3.0
_CA = 3.0
_TR = 0.5

_A4 = 0.51747906167389938633
_A5 = 0.50840057924226870746


def _polylog2(x):
    x_arr = jnp.asarray(x)
    return spence(1.0 - x_arr)


def _normalize_mq(mq):
    if mq is None:
        return []
    out = []
    for item in mq:
        if item is None:
            continue
        if hasattr(item, "first") and hasattr(item, "second"):
            out.append((float(item.first), float(item.second)))
        else:
            pair = list(item)
            if len(pair) != 2:
                raise ValueError("mq entries must be (mass, scale)")
            out.append((float(pair[0]), float(pair[1])))
        if len(out) >= 4:
            break
    return out


def _mass_coeffs(nf: int) -> tuple[list[float], list[float]]:
    beta = [beta0(nf), beta1(nf), beta2(nf), beta3(nf), beta4(nf)]
    gamma = [gamma0(nf), gamma1(nf), gamma2(nf), gamma3(nf), gamma4(nf)]
    b0 = beta[0]
    b = [b_i / b0 for b_i in beta]
    c = [g_i / b0 for g_i in gamma]
    return b, c


def _fsetcx(x: float, loops: int, nf: int) -> float:
    b, c = _mass_coeffs(nf)
    b1, b2, b3, b4 = b[1], b[2], b[3], b[4]
    c0, c1, c2, c3, c4 = c[0], c[1], c[2], c[3], c[4]

    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = (c1 - b1 * c0) * x
    sum_terms[2] = 0.5 * (
        (c1 - b1 * c0) ** 2
        + c2
        - b1 * c1
        + b1 * b1 * c0
        - b2 * c0
    ) * (x * x)
    sum_terms[3] = (
        (c1 - b1 * c0) ** 3 / 6.0
        + 0.5
        * (c1 - b1 * c0)
        * (c2 - b1 * c1 + b1 * b1 * c0 - b2 * c0)
        + (c3 - b1 * c2 + b1 * b1 * c1 - b2 * c1 - b1**3 * c0 + 2.0 * b1 * b2 * c0 - b3 * c0)
        / 3.0
    ) * (x * x * x)
    sum_terms[4] = (
        (c1 - b1 * c0) ** 4 / 24.0
        + (c1 - b1 * c0) ** 2
        * (c2 / 2.0 - b1 * c1 / 2.0 + b1 * b1 * c0 / 2.0 - b2 * c0 / 2.0)
        / 2.0
        + (c2 / 2.0 - b1 * c1 / 2.0 + b1 * b1 * c0 / 2.0 - b2 * c0 / 2.0) ** 2 / 2.0
        + (c1 - b1 * c0)
        * (c3 - b1 * c2 + b1 * b1 * c1 - b2 * c1 - b1**3 * c0 + 2.0 * b1 * b2 * c0 - b3 * c0)
        / 3.0
        + b1**4 * c0 / 4.0
        - 3.0 * b1 * b1 * b2 * c0 / 4.0
        + b2 * b2 * c0 / 4.0
        + b1 * b3 * c0 / 2.0
        - b4 * c0 / 4.0
        - b1**3 * c1 / 4.0
        + b1 * b2 * c1 / 2.0
        - b3 * c1 / 4.0
        + b1 * b1 * c2 / 4.0
        - b2 * c2 / 4.0
        - b1 * c3 / 4.0
        + c4 / 4.0
    ) * (x * x * x * x)

    erg = 0.0
    for i in range(loops):
        erg += sum_terms[i]
    return (x**c0) * erg


def _fMsFromOs1(mu: float, m_os: float) -> float:
    lmM = math.log((mu * mu) / (m_os * m_os))
    return -_CF - (3.0 * _CF * lmM) / 4.0


def _fMsFromOs2(mu: float, m_os: float, nl: float) -> float:
    lmM = math.log((mu * mu) / (m_os * m_os))
    return (
        (-1111.0 * _CA * _CF) / 384.0
        + (7.0 * _CF * _CF) / 128.0
        - (185.0 * _CA * _CF * lmM) / 96.0
        + (21.0 * _CF * _CF * lmM) / 32.0
        - (11.0 * _CA * _CF * lmM * lmM) / 32.0
        + (9.0 * _CF * _CF * lmM * lmM) / 32.0
        + (143.0 * _CF * _TR) / 96.0
        + (13.0 * _CF * lmM * _TR) / 24.0
        + (_CF * lmM * lmM * _TR) / 8.0
        + (71.0 * _CF * nl * _TR) / 96.0
        + (13.0 * _CF * lmM * nl * _TR) / 24.0
        + (_CF * lmM * lmM * nl * _TR) / 8.0
        + (_CA * _CF * _ZETA2) / 2.0
        - (15.0 * _CF * _CF * _ZETA2) / 8.0
        - (3.0 * _CA * _CF * _LOG2 * _ZETA2) / 2.0
        + 3.0 * _CF * _CF * _LOG2 * _ZETA2
        - _CF * _TR * _ZETA2
        + (_CF * nl * _TR * _ZETA2) / 2.0
        + (3.0 * _CA * _CF * _ZETA3) / 8.0
        - (3.0 * _CF * _CF * _ZETA3) / 4.0
    )


def _fMsFromOs3(mu: float, m_os: float, nl: float) -> float:
    lmM = math.log((mu * mu) / (m_os * m_os))
    return (
        (lmM * lmM * (
            -2341.0 * _CA * _CA * _CF
            + 1962.0 * _CA * _CF * _CF
            - 243.0 * _CF * _CF * _CF
            + 1492.0 * _CA * _CF * _TR
            - 468.0 * _CF * _CF * _TR
            + 1492.0 * _CA * _CF * nl * _TR
            - 468.0 * _CF * _CF * nl * _TR
            - 208.0 * _CF * _TR * _TR
            - 416.0 * _CF * nl * _TR * _TR
            - 208.0 * _CF * nl * nl * _TR * _TR
        ))
        / 1152.0
        + (lmM**3 * (
            -242.0 * _CA * _CA * _CF
            + 297.0 * _CA * _CF * _CF
            - 81.0 * _CF * _CF * _CF
            + 176.0 * _CA * _CF * _TR
            - 108.0 * _CF * _CF * _TR
            + 176.0 * _CA * _CF * nl * _TR
            - 108.0 * _CF * _CF * nl * _TR
            - 32.0 * _CF * _TR * _TR
            - 64.0 * _CF * nl * _TR * _TR
            - 32.0 * _CF * nl * nl * _TR * _TR
        ))
        / 1152.0
        + (lmM * (
            -105944.0 * _CA * _CA * _CF
            + 52317.0 * _CA * _CF * _CF
            - 13203.0 * _CF * _CF * _CF
            + 74624.0 * _CA * _CF * _TR
            - 5436.0 * _CF * _CF * _TR
            + 55616.0 * _CA * _CF * nl * _TR
            + 2340.0 * _CF * _CF * nl * _TR
            - 12608.0 * _CF * _TR * _TR
            - 18304.0 * _CF * nl * _TR * _TR
            - 5696.0 * _CF * nl * nl * _TR * _TR
            + 12672.0 * _CA * _CA * _CF * _ZETA2
            - 52704.0 * _CA * _CF * _CF * _ZETA2
            + 19440.0 * _CF * _CF * _CF * _ZETA2
            - 38016.0 * _CA * _CA * _CF * _LOG2 * _ZETA2
            + 91584.0 * _CA * _CF * _CF * _LOG2 * _ZETA2
            - 31104.0 * _CF * _CF * _CF * _LOG2 * _ZETA2
            - 29952.0 * _CA * _CF * _TR * _ZETA2
            + 27648.0 * _CF * _CF * _TR * _ZETA2
            + 13824.0 * _CA * _CF * _LOG2 * _TR * _ZETA2
            - 27648.0 * _CF * _CF * _LOG2 * _TR * _ZETA2
            + 8064.0 * _CA * _CF * nl * _TR * _ZETA2
            + 12096.0 * _CF * _CF * nl * _TR * _ZETA2
            + 13824.0 * _CA * _CF * _LOG2 * nl * _TR * _ZETA2
            - 27648.0 * _CF * _CF * _LOG2 * nl * _TR * _ZETA2
            + 9216.0 * _CF * _TR * _TR * _ZETA2
            + 4608.0 * _CF * nl * _TR * _TR * _ZETA2
            - 4608.0 * _CF * nl * nl * _TR * _TR * _ZETA2
            + 9504.0 * _CA * _CA * _CF * _ZETA3
            - 22896.0 * _CA * _CF * _CF * _ZETA3
            + 7776.0 * _CF * _CF * _CF * _ZETA3
            + 6912.0 * _CA * _CF * _TR * _ZETA3
            - 3456.0 * _CF * _CF * _TR * _ZETA3
            + 6912.0 * _CA * _CF * nl * _TR * _ZETA3
            - 3456.0 * _CF * _CF * nl * _TR * _ZETA3
        ))
        / 13824.0
    )


def _fZmM(nl: float) -> float:
    return (
        -9478333.0 / 93312.0
        + 55.0 * _LOG2**4 / 162.0
        + (-644201.0 / 6480.0 + 587.0 * _LOG2 / 27.0 + 44.0 * _LOG2 * _LOG2 / 27.0) * _ZETA2
        - 61.0 * _ZETA3 / 27.0
        + 3475.0 * _ZETA4 / 432.0
        + 1439.0 * _ZETA2 * _ZETA3 / 72.0
        - 1975.0 * _ZETA5 / 216.0
        + 220.0 * _A4 / 27.0
        + nl * (
            246643.0 / 23328.0
            - _LOG2**4 / 81.0
            + (967.0 / 108.0 + 22.0 * _LOG2 / 27.0 - 4.0 * _LOG2 * _LOG2 / 27.0) * _ZETA2
            + 241.0 * _ZETA3 / 72.0
            - 305.0 * _ZETA4 / 108.0
            - 8.0 * _A4 / 27.0
        )
        + nl * nl * (-2353.0 / 23328.0 - 13.0 * _ZETA2 / 54.0 - 7.0 * _ZETA3 / 54.0)
    )


def _fMsFromOs4(mu: float, m_os: float, nl: float, err: float) -> float:
    lmM = math.log((mu * mu) / (m_os * m_os))
    return (
        -3654.15040757339 * err
        - 1524.2292266911543 * lmM
        - 288.778291935394 * lmM * lmM
        - 32.54735725308642 * lmM**3
        - 1.85546875 * lmM**4
        + nl * nl * nl * (
            0.678141025604516 * err
            + 0.3205521521864135 * lmM
            + 0.0800290327210927 * lmM * lmM
            + 0.010030864197530864 * lmM**3
            + 0.001157407407407407 * lmM**4
        )
        + nl * nl * (
            -43.48241924867489 * err
            - 19.82672048099557 * lmM
            - 4.482957520194182 * lmM * lmM
            - 0.5270061728395061 * lmM**3
            - 0.04108796296296297 * lmM**4
        )
        + nl * (
            756.9421565599532 * err
            + 330.1770776731065 * lmM
            + 67.99849534415492 * lmM * lmM
            + 7.595293209876542 * lmM**3
            + 0.48119212962962954 * lmM**4
        )
    )


def _fZmInvM(nl: float) -> float:
    return (
        8481925.0 / 93312.0
        + (137.0 * nl) / 216.0
        + (652841.0 * _PI * _PI) / 38880.0
        - (nl * _PI * _PI) / 27.0
        - (695.0 * _PI**4) / 7776.0
        - (575.0 * _PI * _PI * _LOG2) / 162.0
        - (22.0 * _PI * _PI * _LOG2 * _LOG2) / 81.0
        - (55.0 * _LOG2**4) / 162.0
        - (220.0 * _A4) / 27.0
        - nl * nl * (-2353.0 / 23328.0 - (13.0 * _PI * _PI) / 324.0 - (7.0 * _ZETA3) / 54.0)
        + (58.0 * _ZETA3) / 27.0
        - (1439.0 * _PI * _PI * _ZETA3) / 432.0
        - nl * (
            246643.0 / 23328.0
            + (967.0 * _PI * _PI) / 648.0
            - (61.0 * _PI**4) / 1944.0
            + (11.0 * _PI * _PI * _LOG2) / 81.0
            - (2.0 * _PI * _PI * _LOG2 * _LOG2) / 81.0
            - _LOG2**4 / 81.0
            - (8.0 * _A4) / 27.0
            + (241.0 * _ZETA3) / 72.0
        )
        + (1975.0 * _ZETA5) / 216.0
    )


def _fOsFromMs1(mu: float, m_ms: float) -> float:
    lmM = math.log((mu * mu) / (m_ms * m_ms))
    return _CF + (3.0 * _CF * lmM) / 4.0


def _fOsFromMs2(mu: float, m_ms: float, nl: float) -> float:
    lmM = math.log((mu * mu) / (m_ms * m_ms))
    return (
        (1111.0 * _CA * _CF) / 384.0
        - (71.0 * _CF * _CF) / 128.0
        - (143.0 * _CF * _TR) / 96.0
        - (71.0 * _CF * nl * _TR) / 96.0
        + lmM * ((185.0 * _CA * _CF) / 96.0 - (9.0 * _CF * _CF) / 32.0 - (13.0 * _CF * _TR) / 24.0 - (13.0 * _CF * nl * _TR) / 24.0)
        + lmM * lmM * ((11.0 * _CA * _CF) / 32.0 + (9.0 * _CF * _CF) / 32.0 - (_CF * _TR) / 8.0 - (_CF * nl * _TR) / 8.0)
        - (_CA * _CF * _ZETA2) / 2.0
        + (15.0 * _CF * _CF * _ZETA2) / 8.0
        + (3.0 * _CA * _CF * _LOG2 * _ZETA2) / 2.0
        - 3.0 * _CF * _CF * _LOG2 * _ZETA2
        + _CF * _TR * _ZETA2
        - (_CF * nl * _TR * _ZETA2) / 2.0
        - (3.0 * _CA * _CF * _ZETA3) / 8.0
        + (3.0 * _CF * _CF * _ZETA3) / 4.0
    )


def _fOsFromMs3(mu: float, m_ms: float, nl: float) -> float:
    lmM = math.log((mu * mu) / (m_ms * m_ms))
    return (
        lmM**3 * (
            (121.0 * _CA * _CA * _CF) / 576.0
            + (33.0 * _CA * _CF * _CF) / 128.0
            + (9.0 * _CF * _CF * _CF) / 128.0
            - (11.0 * _CA * _CF * _TR) / 72.0
            - (3.0 * _CF * _CF * _TR) / 32.0
            - (11.0 * _CA * _CF * nl * _TR) / 72.0
            - (3.0 * _CF * _CF * nl * _TR) / 32.0
            + (_CF * _TR * _TR) / 36.0
            + (_CF * nl * _TR * _TR) / 18.0
            + (_CF * nl * nl * _TR * _TR) / 36.0
        )
        + lmM * lmM * (
            (2341.0 * _CA * _CA * _CF) / 1152.0
            + (21.0 * _CA * _CF * _CF) / 64.0
            - (63.0 * _CF * _CF * _CF) / 128.0
            - (373.0 * _CA * _CF * _TR) / 288.0
            - (3.0 * _CF * _CF * _TR) / 32.0
            - (373.0 * _CA * _CF * nl * _TR) / 288.0
            - (3.0 * _CF * _CF * nl * _TR) / 32.0
            + (13.0 * _CF * _TR * _TR) / 72.0
            + (13.0 * _CF * nl * _TR * _TR) / 36.0
            + (13.0 * _CF * nl * nl * _TR * _TR) / 72.0
        )
        + lmM * (
            (13243.0 * _CA * _CA * _CF) / 1728.0
            - (4219.0 * _CA * _CF * _CF) / 1536.0
            + (495.0 * _CF * _CF * _CF) / 512.0
            - (583.0 * _CA * _CF * _TR) / 108.0
            - (307.0 * _CF * _CF * _TR) / 384.0
            - (869.0 * _CA * _CF * nl * _TR) / 216.0
            - (91.0 * _CF * _CF * nl * _TR) / 384.0
            + (197.0 * _CF * _TR * _TR) / 216.0
            + (143.0 * _CF * nl * _TR * _TR) / 108.0
            + (89.0 * _CF * nl * nl * _TR * _TR) / 216.0
            - (11.0 * _CA * _CA * _CF * _ZETA2) / 12.0
            + (49.0 * _CA * _CF * _CF * _ZETA2) / 16.0
            + (45.0 * _CF * _CF * _CF * _ZETA2) / 32.0
            + (11.0 * _CA * _CA * _CF * _LOG2 * _ZETA2) / 4.0
            - (35.0 * _CA * _CF * _CF * _LOG2 * _ZETA2) / 8.0
            - (9.0 * _CF * _CF * _CF * _LOG2 * _ZETA2) / 4.0
            + (13.0 * _CA * _CF * _TR * _ZETA2) / 6.0
            - (_CF * _CF * _TR * _ZETA2) / 2.0
            - _CA * _CF * _LOG2 * _TR * _ZETA2
            + 2.0 * _CF * _CF * _LOG2 * _TR * _ZETA2
            - (7.0 * _CA * _CF * nl * _TR * _ZETA2) / 12.0
            - (13.0 * _CF * _CF * nl * _TR * _ZETA2) / 8.0
            - _CA * _CF * _LOG2 * nl * _TR * _ZETA2
            + 2.0 * _CF * _CF * _LOG2 * nl * _TR * _ZETA2
            - (2.0 * _CF * _TR * _TR * _ZETA2) / 3.0
            - (_CF * nl * _TR * _TR * _ZETA2) / 3.0
            + (_CF * nl * nl * _TR * _TR * _ZETA2) / 3.0
            - (11.0 * _CA * _CA * _CF * _ZETA3) / 16.0
            + (35.0 * _CA * _CF * _CF * _ZETA3) / 32.0
            + (9.0 * _CF * _CF * _CF * _ZETA3) / 16.0
            - (_CA * _CF * _TR * _ZETA3) / 2.0
            + (_CF * _CF * _TR * _ZETA3) / 4.0
            - (_CA * _CF * nl * _TR * _ZETA3) / 2.0
            + (_CF * _CF * nl * _TR * _ZETA3) / 4.0
        )
    )


def _fOsFromMs4(mu: float, m_ms: float, nl: float, err: float) -> float:
    lmM = math.log((mu * mu) / (m_ms * m_ms))
    return (
        3567.602784989066 * err
        + 1727.2260148986106 * lmM
        + 409.2429990574718 * lmM * lmM
        + 66.93663194444443 * lmM**3
        + 8.056278935185185 * lmM**4
        + nl * nl * nl * (
            -0.678141025604516 * err
            - 0.3205521521864134 * lmM
            - 0.0800290327210927 * lmM * lmM
            - 0.010030864197530864 * lmM**3
            - 0.001157407407407407 * lmM**4
        )
        + nl * (
            -745.7207145811878 * err
            - 358.29765085086774 * lmM
            - 87.39262571554698 * lmM * lmM
            - 11.883873456790122 * lmM**3
            - 1.2705439814814814 * lmM**4
        )
        + nl * nl * (
            43.396250117985666 * err
            + 20.528466368867228 * lmM
            + 4.971905254812516 * lmM * lmM
            + 0.6304012345679011 * lmM**3
            + 0.06655092592592593 * lmM**4
        )
    )


def _deltamOS2mMS(m_os: float, mq, asmu: float, mu: float, nlq: int, loops: int) -> float:
    mq_list = _normalize_mq(mq)
    if not mq_list:
        return 0.0
    erg = 0.0
    for i, (mqi, muf) in enumerate(mq_list):
        x = mqi / m_os
        nl = nlq - i
        if loops >= 2:
            erg += asmu * asmu * 2.0 / 3.0 * (
                48.0 * x**4 * math.log(x) ** 2
                + 48.0 * x * x * math.log(x)
                + 72.0 * x * x
                + 8.0 * _PI * _PI * (x**4 - 3.0 * x**3 - 3.0 * x)
                - 48.0 * (x + 1.0) ** 2 * (x * x - x + 1.0) * (math.log(x) * math.log(x + 1.0) + _polylog2(-x))
                - 48.0 * (x - 1.0) ** 2 * (x * x + x + 1.0) * (math.log(x) * math.log(1.0 - x) + _polylog2(x))
            ) / 96.0
        if loops >= 3:
            erg += asmu**3 * (
                -21.8714 * x
                - 4.348 * x * x
                - 1.02211 * x**3
                - 0.0493333 * x**4
                + nl * (0.982667 * x + 0.300333 * x * x)
                + math.log(mu * mu / (m_os * m_os))
                * (-6.61056 * x + 2.46511 * x * x - 0.724333 * x**3 + nl * (0.534667 * x - 0.22 * x * x + 0.067 * x**3))
                + math.log(x) * (16.9477 * x - 1.10133 * nl * x + 2.78756 * x * x - 0.0343333 * x**3)
                + 8.0
                / 9.0
                * (3.0 / 2.0 * math.log(muf * muf / (mqi * mqi)) + 2.0)
                * x
                / 24.0
                * (
                    24.0 * x**3 * math.log(x) ** 2
                    + 12.0 * x * math.log(x)
                    + 24.0 * x
                    + _PI * _PI * (4.0 * x**3 - 9.0 * x * x - 3.0)
                    - 6.0 * (4.0 * x**3 + 3.0 * x * x + 1.0) * (math.log(x) * math.log(x + 1.0) + _polylog2(-x))
                    - 6.0 * (x - 1.0) * (4.0 * x * x + x + 1.0) * (math.log(x) * math.log(1.0 - x) + _polylog2(x))
                )
            )
    return erg


def _deltamMS2mOS(m_ms: float, mq, asmu: float, mu: float, nlq: int, loops: int) -> float:
    mq_list = _normalize_mq(mq)
    if not mq_list:
        return 0.0
    lmu = math.log(mu * mu / (m_ms * m_ms))
    erg = 0.0
    for i, (mqi, muf) in enumerate(mq_list):
        x = mqi / m_ms
        nl = nlq - i
        if loops >= 2:
            erg += -asmu * asmu * 2.0 / 3.0 * (
                48.0 * x**4 * math.log(x) ** 2
                + 48.0 * x * x * math.log(x)
                + 72.0 * x * x
                + 8.0 * _PI * _PI * (x**4 - 3.0 * x**3 - 3.0 * x)
                - 48.0 * (x + 1.0) ** 2 * (x * x - x + 1.0) * (math.log(x) * math.log(x + 1.0) + _polylog2(-x))
                - 48.0 * (x - 1.0) ** 2 * (x * x + x + 1.0) * (math.log(x) * math.log(1.0 - x) + _polylog2(x))
            ) / 96.0
        if loops >= 3:
            term1 = (
                -21.8714 * x
                - 4.348 * x * x
                - 1.02211 * x**3
                - 0.0493333 * x**4
                + nl * (0.982667 * x + 0.300333 * x * x)
                + lmu * (-6.61056 * x + 2.46511 * x * x - 0.724333 * x**3 + nl * (0.534667 * x - 0.22 * x * x + 0.067 * x**3))
                + math.log(x) * (16.9477 * x - 1.10133 * nl * x + 2.78756 * x * x - 0.0343333 * x**3)
                + 8.0
                / 9.0
                * (3.0 / 2.0 * math.log(muf * muf / (mqi * mqi)) + 2.0)
                * x
                / 24.0
                * (
                    24.0 * x**3 * math.log(x) ** 2
                    + 12.0 * x * math.log(x)
                    + 24.0 * x
                    + _PI * _PI * (4.0 * x**3 - 9.0 * x * x - 3.0)
                    - 6.0 * (4.0 * x**3 + 3.0 * x * x + 1.0) * (math.log(x) * math.log(x + 1.0) + _polylog2(-x))
                    - 6.0 * (x - 1.0) * (4.0 * x * x + x + 1.0) * (math.log(x) * math.log(1.0 - x) + _polylog2(x))
                )
            )
            term2 = (
                48.0 * (3.0 * lmu + 7.0) * x**4 * math.log(x) ** 2
                + 144.0 * x * x * math.log(x)
                + 312.0 * x * x
                + 8.0 * _PI * _PI * (7.0 * x**4 - 15.0 * x**3 - 3.0 * x)
                - lmu * (-72.0 * x * x + 12.0 * _PI * _PI * (-2.0 * x**4 + 3.0 * x**3 - 3.0 * x))
                - 48.0
                * (
                    (7.0 * x + 5.0) * x**3
                    + x
                    + 1.5 * lmu * (2.0 * x**4 + x**3 - x - 2.0)
                    - 1.0
                )
                * (math.log(x) * math.log(x + 1.0) + _polylog2(-x))
                - 48.0
                * (x - 1.0)
                * (
                    1.5 * lmu * (2.0 * x**3 + x * x + x + 2.0)
                    + 7.0 * x**3
                    + 2.0 * x * x
                    + 2.0 * x
                    + 1.0
                )
                * (math.log(x) * math.log(1.0 - x) + _polylog2(x))
            )
            erg += -asmu**3 * term1 + asmu**3 * 8.0 / 9.0 * term2 / 192.0
    return erg


def _fmq6to5os(api: float, mass: float, mu: float, nlq: float, loops: int) -> float:
    lmM = math.log((mu * mu) / (mass * mass))
    if loops == 1:
        return 1.0
    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = 0.0
    sum_terms[2] = -api * api * (89.0 / 432.0 - (5.0 * lmM) / 36.0 + lmM * lmM / 12.0)
    sum_terms[3] = -api**3 * (
        1871.0 / 2916.0
        - (4.0 * _A4) / 9.0
        - _LOG2**4 / 54.0
        + (299.0 * lmM) / 2592.0
        + (299.0 * lmM * lmM) / 432.0
        + (35.0 * lmM**3) / 216.0
        + (1327.0 * nlq) / 11664.0
        - (53.0 * lmM * nlq) / 432.0
        - (lmM**3 * nlq) / 108.0
        + (_LOG2 * _LOG2 * _ZETA2) / 9.0
        - (407.0 * _ZETA3) / 864.0
        - (5.0 * lmM * _ZETA3) / 6.0
        - (2.0 * nlq * _ZETA3) / 27.0
        + (103.0 * _ZETA4) / 72.0
    )
    sum_terms[4] = api**4 * (
        -122722873709.0 / 3292047360.0
        - (6941.0 * lmM**3) / 2592.0
        - (1147.0 * lmM**4) / 3456.0
        - (16187201.0 * _PI**4) / 52254720.0
        + (787.0 * _PI**6) / 81648.0
        + (5.0 * _ZETA2) / 9.0
        + (5.0 * _LOG2 * _ZETA2) / 27.0
        - (5.0 * _ZETA3) / 108.0
        + (145.0 * _PI**4 * _LOG2) / 1944.0
        - (1924649.0 * _PI**2 * _LOG2 * _LOG2) / 4354560.0
        + (59.0 * _PI**2 * _LOG2**3) / 972.0
        + (1924649.0 * _LOG2**4) / 4354560.0
        - (59.0 * _LOG2**5) / 1620.0
        - (4.0 * _LOG2**6) / 81.0
        + (_LOG2**5 * math.log(16.0)) / 81.0
        + (1924649.0 * _A4) / 181440.0
        + (118.0 * _A5) / 27.0
        + nlq * nlq * (
            -17671.0 / 124416.0
            - (31.0 * lmM * lmM) / 1296.0
            - lmM**4 / 864.0
            + (7.0 * _PI**4) / 8640.0
            + lmM * (3401.0 / 46656.0 - (7.0 * _ZETA3) / 108.0)
            + (5.0 * _ZETA3) / 864.0
        )
        + (443509931.0 * _ZETA3) / 40642560.0
        - (1061.0 * _ZETA3 * _ZETA3) / 576.0
        + lmM * lmM * (-34297.0 / 3456.0 + (175.0 * _ZETA3) / 48.0)
        + lmM
        * (
            99319.0 / 41472.0
            - (481.0 * _PI**4) / 2880.0
            - (2.0 * _ZETA2) / 3.0
            - (2.0 * _LOG2 * _ZETA2) / 9.0
            + _ZETA3 / 18.0
            - (11.0 * _PI**2 * _LOG2 * _LOG2) / 72.0
            + (11.0 * _LOG2**4) / 72.0
            + (11.0 * _A4) / 3.0
            + (47317.0 * _ZETA3) / 3072.0
            - (575.0 * _ZETA5) / 72.0
        )
        + nlq
        * (
            2403419.0 / 746496.0
            + (10237.0 * lmM * lmM) / 10368.0
            + (47.0 * lmM**3) / 288.0
            + (17.0 * lmM**4) / 432.0
            + (245.0 * _PI**4) / 62208.0
            - (5.0 * _ZETA2) / 54.0
            - (49.0 * _PI**4 * _LOG2) / 6480.0
            + (49.0 * _PI**2 * _LOG2 * _LOG2) / 2592.0
            - (_PI**2 * _LOG2**3) / 162.0
            - (49.0 * _LOG2**4) / 2592.0
            + _LOG2**5 / 270.0
            - (49.0 * _A4) / 108.0
            - (4.0 * _A5) / 9.0
            + lmM
            * (
                -26443.0 / 93312.0
                + (163.0 * _PI**4) / 12960.0
                + _ZETA2 / 9.0
                + (_PI**2 * _LOG2 * _LOG2) / 108.0
                - _LOG2**4 / 108.0
                - (2.0 * _A4) / 9.0
                - (599.0 * _ZETA3) / 1728.0
            )
            + (1075.0 * _ZETA3) / 1728.0
            - (497.0 * _ZETA5) / 288.0
        )
        + (59015.0 * _ZETA5) / 1728.0
    )
    total = 0.0
    for i in range(loops):
        total += sum_terms[i]
    return total


def _fmq5to6os(api: float, mass: float, mu: float, nlq: float, loops: int) -> float:
    lmM = math.log((mu * mu) / (mass * mass))
    if loops == 1:
        return 1.0
    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = 0.0
    sum_terms[2] = api * api * (89.0 / 432.0 - (5.0 * lmM) / 36.0 + lmM * lmM / 12.0)
    sum_terms[3] = api**3 * (
        1871.0 / 2916.0
        - (4.0 * _A4) / 9.0
        - _LOG2**4 / 54.0
        + (121.0 * lmM) / 2592.0
        + (319.0 * lmM * lmM) / 432.0
        + (29.0 * lmM**3) / 216.0
        + (1327.0 * nlq) / 11664.0
        - (53.0 * lmM * nlq) / 432.0
        - (lmM**3 * nlq) / 108.0
        + (_LOG2 * _LOG2 * _ZETA2) / 9.0
        - (407.0 * _ZETA3) / 864.0
        - (5.0 * lmM * _ZETA3) / 6.0
        - (2.0 * nlq * _ZETA3) / 27.0
        + (103.0 * _ZETA4) / 72.0
    )
    sum_terms[4] = api**4 * (
        122466970229.0 / 3292047360.0
        - (1924649.0 * _A4) / 181440.0
        - (118.0 * _A5) / 27.0
        - (1924649.0 * _LOG2**4) / 4354560.0
        + (59.0 * _LOG2**5) / 1620.0
        - (1126487.0 * lmM) / 373248.0
        - (31.0 * _A4 * lmM) / 9.0
        - (31.0 * _LOG2**4 * lmM) / 216.0
        + (104803.0 * lmM * lmM) / 10368.0
        + (1403.0 * lmM**3) / 648.0
        + (305.0 * lmM**4) / 1152.0
        - (2403419.0 * nlq) / 746496.0
        + (49.0 * _A4 * nlq) / 108.0
        + (4.0 * _A5 * nlq) / 9.0
        + (49.0 * _LOG2**4 * nlq) / 2592.0
        - (_LOG2**5 * nlq) / 270.0
        + (7045.0 * lmM * nlq) / 31104.0
        + (2.0 * _A4 * lmM * nlq) / 9.0
        + (_LOG2**4 * lmM * nlq) / 108.0
        - (9601.0 * lmM * lmM * nlq) / 10368.0
        - (47.0 * lmM**3 * nlq) / 288.0
        - (5.0 * lmM**4 * nlq) / 144.0
        + (17671.0 * nlq * nlq) / 124416.0
        - (3401.0 * lmM * nlq * nlq) / 46656.0
        + (31.0 * lmM * lmM * nlq * nlq) / 1296.0
        + (lmM**4 * nlq * nlq) / 864.0
        - (5.0 * _ZETA2) / 9.0
        - (5.0 * _LOG2 * _ZETA2) / 27.0
        + (1924649.0 * _LOG2 * _LOG2 * _ZETA2) / 725760.0
        - (59.0 * _LOG2**3 * _ZETA2) / 162.0
        + (2.0 * lmM * _ZETA2) / 3.0
        + (2.0 * _LOG2 * lmM * _ZETA2) / 9.0
        + (31.0 * _LOG2 * _LOG2 * lmM * _ZETA2) / 36.0
        + (5.0 * nlq * _ZETA2) / 54.0
        - (49.0 * _LOG2 * _LOG2 * nlq * _ZETA2) / 432.0
        + (_LOG2**3 * nlq * _ZETA2) / 27.0
        - (lmM * nlq * _ZETA2) / 9.0
        - (_LOG2 * _LOG2 * lmM * nlq * _ZETA2) / 18.0
        - (441628331.0 * _ZETA3) / 40642560.0
        - (420877.0 * lmM * _ZETA3) / 27648.0
        - (155.0 * lmM * lmM * _ZETA3) / 48.0
        - (1075.0 * nlq * _ZETA3) / 1728.0
        + (221.0 * lmM * nlq * _ZETA3) / 576.0
        - (5.0 * nlq * nlq * _ZETA3) / 864.0
        + (7.0 * lmM * nlq * nlq * _ZETA3) / 108.0
        + (1061.0 * _ZETA3 * _ZETA3) / 576.0
        + (16187201.0 * _ZETA4) / 580608.0
        - (725.0 * _LOG2 * _ZETA4) / 108.0
        + (4123.0 * lmM * _ZETA4) / 288.0
        - (1225.0 * nlq * _ZETA4) / 3456.0
        + (49.0 * _LOG2 * nlq * _ZETA4) / 72.0
        - (163.0 * lmM * nlq * _ZETA4) / 144.0
        - (7.0 * nlq * nlq * _ZETA4) / 96.0
        - (59015.0 * _ZETA5) / 1728.0
        + (575.0 * lmM * _ZETA5) / 72.0
        + (497.0 * nlq * _ZETA5) / 288.0
        - (3935.0 * _ZETA6) / 432.0
    )
    total = 0.0
    for i in range(loops):
        total += sum_terms[i]
    return total


def DecMqUpOS(mq: float, alphas: float, massth: float, muth: float, nf: int, loops: int) -> float:
    if not 1 <= loops <= 5:
        raise ValueError(f"DecMqUpOS: loops must be 1..5, got {loops}")
    api = alphas / _PI
    return mq * _fmq6to5os(api, massth, muth, float(nf), loops)


def DecMqDownOS(mq: float, alphas: float, massth: float, muth: float, nf: int, loops: int) -> float:
    if not 1 <= loops <= 5:
        raise ValueError(f"DecMqDownOS: loops must be 1..5, got {loops}")
    api = alphas / _PI
    return mq * _fmq5to6os(api, massth, muth, float(nf), loops)


def mMS2mMS(mmu0: float, alphas0: float, alphas1: float, nf: int, loops: int) -> float:
    r"""Run the MS-bar mass between two scales using αs at those scales.

    Parameters
    ----------
    mmu0 : float
        $m_{\overline{\mathrm{MS}}}(\mu_0)$.
    alphas0 : float
        $\alpha_s^{(n_f)}(\mu_0)$.
    alphas1 : float
        $\alpha_s^{(n_f)}(\mu_1)$.
    nf : int
        Number of active flavours.
    loops : int
        Loop order (0..5).
    """
    if loops < 0 or loops > 5:
        raise ValueError(f"mMS2mMS only supports loops=0..5 (got {loops}).")
    if loops == 0:
        return mmu0

    x0 = alphas0 / math.pi
    x1 = alphas1 / math.pi
    c0 = _fsetcx(x0, loops, nf)
    c1 = _fsetcx(x1, loops, nf)
    return mmu0 * c1 / c0


def mL2mH(
    mql: float,
    alphas_low: float,
    mu1: float,
    decpar: Iterable[object],
    mu2: float,
    loops: int,
) -> float:
    """Run MS-bar mass from low to high scale with decoupling at thresholds."""
    if not 1 <= loops <= 5:
        raise ValueError(f"mL2mH: loops must be 1..5, got {loops}")

    entries = _normalize_decpar(decpar)
    if not entries:
        raise ValueError("decpar must contain at least one threshold")

    entries = sorted(entries, key=lambda d: d.nf)
    _check_consecutive_nfs([d.nf for d in entries], direction=1)

    asini = alphas_low
    muini = mu1
    mqini = mql
    for dec in entries:
        as_at_muth = AlphasExact(asini, muini, dec.muth, nf=dec.nf - 1, loops=loops)
        mq_at_muth = mMS2mMS(mqini, asini, as_at_muth, nf=dec.nf - 1, loops=loops)
        asini = DecAsUpOS(as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        mqini = DecMqUpOS(mq_at_muth, as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        muini = dec.muth

    as_final = AlphasExact(asini, muini, mu2, nf=entries[-1].nf, loops=loops)
    return mMS2mMS(mqini, asini, as_final, nf=entries[-1].nf, loops=loops)


def mH2mL(
    mqh: float,
    alphas_high: float,
    mu1: float,
    decpar: Iterable[object],
    mu2: float,
    loops: int,
) -> float:
    """Run MS-bar mass from high to low scale with decoupling at thresholds."""
    if not 1 <= loops <= 5:
        raise ValueError(f"mH2mL: loops must be 1..5, got {loops}")

    entries = _normalize_decpar(decpar)
    if not entries:
        raise ValueError("decpar must contain at least one threshold")

    entries = sorted(entries, key=lambda d: d.nf, reverse=True)
    _check_consecutive_nfs([d.nf for d in entries], direction=-1)

    asini = alphas_high
    muini = mu1
    mqini = mqh
    for dec in entries:
        as_at_muth = AlphasExact(asini, muini, dec.muth, nf=dec.nf, loops=loops)
        mq_at_muth = mMS2mMS(mqini, asini, as_at_muth, nf=dec.nf, loops=loops)
        asini = DecAsDownOS(as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        mqini = DecMqDownOS(mq_at_muth, as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        muini = dec.muth

    as_final = AlphasExact(asini, muini, mu2, nf=entries[-1].nf - 1, loops=loops)
    return mMS2mMS(mqini, asini, as_final, nf=entries[-1].nf - 1, loops=loops)


def mOS2mMS(
    m_os: float,
    mq,
    alphas: float,
    mu: float,
    nf: int,
    loops: int,
    fdelm: float = 1.0,
) -> float:
    if loops < 0 or loops > 4:
        raise ValueError(f"mOS2mMS: loops must be 0..4, got {loops}")
    if loops == 4 and (nf < 4 or nf > 6):
        raise ValueError("mOS2mMS: 4-loop result only implemented for nf=4..6")

    deltalight = _deltamOS2mMS(m_os, mq, alphas / _PI, mu, nf - 1, loops)

    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = alphas * _fMsFromOs1(mu, m_os) / _PI
    sum_terms[2] = alphas * alphas * _fMsFromOs2(mu, m_os, nf - 1) / (_PI * _PI)
    sum_terms[3] = (
        alphas**3
        * (_fMsFromOs3(mu, m_os, nf - 1) + _fZmM(nf - 1))
        / (_PI**3)
    )
    sum_terms[4] = alphas**4 * _fMsFromOs4(mu, m_os, nf - 1, fdelm) / (_PI**4)

    if loops == 0:
        erg = 1.0
    else:
        erg = 0.0
        for i in range(loops + 1):
            erg += sum_terms[i]

    return m_os * (erg + deltalight)


def mMS2mOS(
    m_ms: float,
    mq,
    alphas: float,
    mu: float,
    nf: int,
    loops: int,
    fdelm: float = 1.0,
) -> float:
    if loops < 0 or loops > 4:
        raise ValueError(f"mMS2mOS: loops must be 0..4, got {loops}")
    if loops == 4 and (nf < 4 or nf > 6):
        raise ValueError("mMS2mOS: 4-loop result only implemented for nf=4..6")

    deltalight = _deltamMS2mOS(m_ms, mq, alphas / _PI, mu, nf - 1, loops)

    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = alphas * _fOsFromMs1(mu, m_ms) / _PI
    sum_terms[2] = alphas * alphas * _fOsFromMs2(mu, m_ms, nf - 1) / (_PI * _PI)
    sum_terms[3] = (
        alphas**3
        * (_fOsFromMs3(mu, m_ms, nf - 1) + _fZmInvM(nf - 1))
        / (_PI**3)
    )
    sum_terms[4] = alphas**4 * _fOsFromMs4(mu, m_ms, nf - 1, fdelm) / (_PI**4)

    if loops == 0:
        erg = 1.0
    else:
        erg = 0.0
        for i in range(loops + 1):
            erg += sum_terms[i]

    return m_ms * (erg + deltalight)
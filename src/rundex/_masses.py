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

try:
    import mpmath as _mp
    _A4 = float(_mp.polylog(4, 0.5))
    _A5 = float(_mp.polylog(5, 0.5))
except ImportError:
    _A4 = 0.51747906167389938633
    _A5 = 0.50840057924226870746


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
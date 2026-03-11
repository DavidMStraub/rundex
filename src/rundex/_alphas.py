"""
Running of the strong coupling αs.

Public functions
----------------
LamExpl(alphas, mu, nf, loops)
    Compute Λ^(nf) from αs(μ) using an explicit closed-form formula.

LamImpl(alphas, mu, nf, loops)
    Compute Λ^(nf) from αs(μ) by numerically inverting AlphasLam.

AlphasLam(lam, mu, nf, loops)
    Compute αs(μ) from Λ^(nf) using the explicit expansion in 1/(b0 L).

AlphasExact(alphas0, mu0, mu, nf, loops)
    Compute αs(μ) by solving the RGE numerically (via diffrax).

Conventions
-----------
* αs ≡ α_s  (not α_s/π).  The internal variable `api` = α_s/π is used
  inside the β-function series.
* All scales (mu, lam, …) are in the same units (GeV by convention).
* `loops` is always a plain Python int.  It is never a JAX-traced value,
  so all if/elif dispatches compile away under jax.jit when nf and loops
  are declared static.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Dopri5,
    diffeqsolve,
)
from scipy.optimize import newton

from ._coeffs import beta0, beta1, beta2, beta3, beta4

# Minimum μ/Λ ratio before a warning is printed (matches Mathematica default)
_RMULAM = 1.5

# ---------------------------------------------------------------------------
# Decoupling constants for αs across heavy thresholds (OS scheme)
# ---------------------------------------------------------------------------

_PI = math.pi
_LOG2 = math.log(2.0)
_ZETA2 = _PI * _PI / 6.0
_ZETA3 = 1.2020569031595942854
_ZETA4 = _PI**4 / 90.0
_ZETA5 = 1.0369277551433699263

try:
    import mpmath as _mp
    _A4 = float(_mp.polylog(4, 0.5))
    _A5 = float(_mp.polylog(5, 0.5))
except ImportError:
    _A4 = 0.51747906167389938633
    _A5 = 0.50840057924226870746


@dataclass(frozen=True)
class Decoupling:
    """Threshold data: number of flavours, pole mass, and decoupling scale."""

    nf: int
    Mth: float
    muth: float


def _normalize_decpar(decpar: Iterable[object]) -> list[Decoupling]:
    entries: list[Decoupling] = []
    for item in decpar:
        if item is None:
            continue
        if isinstance(item, Decoupling):
            dec = item
        elif isinstance(item, dict):
            dec = Decoupling(int(item["nf"]), float(item["Mth"]), float(item["muth"]))
        else:
            nf = getattr(item, "nf", None)
            if nf is not None:
                dec = Decoupling(int(nf), float(getattr(item, "Mth")), float(getattr(item, "muth")))
            else:
                seq = list(item)
                if len(seq) != 3:
                    raise ValueError("decpar entries must have (nf, Mth, muth)")
                dec = Decoupling(int(seq[0]), float(seq[1]), float(seq[2]))
        if dec.nf != 0:
            entries.append(dec)
    return entries


def _fas5to6os(api: float, mass: float, mu: float, nlq: float, loops: int) -> float:
    lmM = math.log((mu * mu) / (mass * mass))
    if loops == 1:
        return 1.0
    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = api * (-lmM) / 6.0
    sum_terms[2] = api**2 * (-7.0 / 24.0 - (19.0 * lmM) / 24.0 + (lmM * lmM) / 36.0)
    sum_terms[3] = api**3 * (
        -58933.0 / 124416.0
        - (8521.0 * lmM) / 1728.0
        - (131.0 * lmM * lmM) / 576.0
        - (lmM * lmM * lmM) / 216.0
        + nlq * (2479.0 / 31104.0 + (409.0 * lmM) / 1728.0 + _ZETA2 / 9.0)
        - (2.0 * _ZETA2) / 3.0
        - (2.0 * _ZETA2 * _LOG2) / 9.0
        - (80507.0 * _ZETA3) / 27648.0
    )
    sum_terms[4] = api**4 * (
        -141841753.0 / 24494400.0
        - (7693.0 * lmM * lmM) / 1152.0
        - (8371.0 * lmM * lmM * lmM) / 10368.0
        + lmM**4 / 1296.0
        - (644201.0 * _PI * _PI) / 116640.0
        - (71102219.0 * _PI**4) / 195955200.0
        - (49.0 * _ZETA2) / 18.0
        - (49.0 * _LOG2 * _ZETA2) / 54.0
        + (49.0 * _ZETA3) / 216.0
        + (587.0 * _PI * _PI * _LOG2) / 486.0
        + (9318467.0 * _PI**4 * _LOG2) / 32659200.0
        - (2913037.0 * _PI * _PI * _LOG2 * _LOG2) / 1306368.0
        + (340853.0 * _PI * _PI * _LOG2**3) / 816480.0
        + (3179149.0 * _LOG2**4) / 1306368.0
        - (340853.0 * _LOG2**5) / 1360800.0
        + (3179149.0 * _A4) / 54432.0
        + (340853.0 * _A5) / 11340.0
        + lmM * (
            -19696909.0 / 746496.0
            - (29.0 * _ZETA2) / 9.0
            - (29.0 * _LOG2 * _ZETA2) / 27.0
            + (59.0 * _ZETA3) / 108.0
            - (2529743.0 * _ZETA3) / 165888.0
        )
        + nlq * nlq * (
            -140825.0 / 1492992.0
            - (493.0 * lmM * lmM) / 20736.0
            - (13.0 * _PI * _PI) / 972.0
            + lmM * (-1679.0 / 186624.0 - _ZETA2 / 27.0)
            - (19.0 * _ZETA3) / 1728.0
        )
        - (2428169183.0 * _ZETA3) / 87091200.0
        + (1439.0 * _PI * _PI * _ZETA3) / 1296.0
        + nlq * (
            1773073.0 / 746496.0
            + (6661.0 * lmM * lmM) / 10368.0
            + (107.0 * lmM * lmM * lmM) / 1728.0
            + (967.0 * _PI * _PI) / 1944.0
            - (697709.0 * _PI**4) / 14929920.0
            + (49.0 * _ZETA2) / 108.0
            + (11.0 * _PI * _PI * _LOG2) / 243.0
            - (1709.0 * _PI * _PI * _LOG2 * _LOG2) / 124416.0
            + (173.0 * _LOG2**4) / 124416.0
            + (173.0 * _A4) / 5184.0
            + (4756441.0 * _ZETA3) / 995328.0
            + lmM * (
                1110443.0 / 373248.0
                + (41.0 * _ZETA2) / 54.0
                + (2.0 * _LOG2 * _ZETA2) / 27.0
                + (7.0 * _ZETA3) / 27.0
                + (110779.0 * _ZETA3) / 82944.0
            )
            + (115.0 * _ZETA5) / 576.0
        )
        - (40596749.0 * _ZETA5) / 1451520.0
    )
    total = 0.0
    for i in range(loops):
        total += sum_terms[i]
    return total


def _fas6to5os(api: float, mass: float, mu: float, nlq: float, loops: int) -> float:
    lmM = math.log((mu * mu) / (mass * mass))
    if loops == 1:
        return 1.0
    sum_terms = [0.0] * 5
    sum_terms[0] = 1.0
    sum_terms[1] = api * (lmM) / 6.0
    sum_terms[2] = api**2 * (7.0 / 24.0 + (19.0 * lmM) / 24.0 + (lmM * lmM) / 36.0)
    sum_terms[3] = api**3 * (
        58933.0 / 124416.0
        + (2.0 * _ZETA2) / 3.0
        + (2.0 * _ZETA2 * _LOG2) / 9.0
        + (80507.0 * _ZETA3) / 27648.0
        + (8941.0 * lmM) / 1728.0
        + (511.0 * lmM * lmM) / 576.0
        + (lmM * lmM * lmM) / 216.0
        + nlq * (-2479.0 / 31104.0 - _ZETA2 / 9.0 - (409.0 * lmM) / 1728.0)
    )
    sum_terms[4] = api**4 * (
        2313952.0 / 382725.0
        + (47039.0 * lmM * lmM) / 3456.0
        + (14149.0 * lmM * lmM * lmM) / 10368.0
        + lmM**4 / 1296.0
        + (644201.0 * _PI * _PI) / 116640.0
        + (71102219.0 * _PI**4) / 195955200.0
        + (49.0 * _ZETA2) / 18.0
        + (49.0 * _LOG2 * _ZETA2) / 54.0
        - (49.0 * _ZETA3) / 216.0
        - (587.0 * _PI * _PI * _LOG2) / 486.0
        - (9318467.0 * _PI**4 * _LOG2) / 32659200.0
        + (2913037.0 * _PI * _PI * _LOG2 * _LOG2) / 1306368.0
        - (340853.0 * _PI * _PI * _LOG2**3) / 816480.0
        - (3179149.0 * _LOG2**4) / 1306368.0
        + (340853.0 * _LOG2**5) / 1360800.0
        - (3179149.0 * _A4) / 54432.0
        - (340853.0 * _A5) / 11340.0
        + nlq * nlq * (
            140825.0 / 1492992.0
            + (493.0 * lmM * lmM) / 20736.0
            + (13.0 * _PI * _PI) / 972.0
            + lmM * (1679.0 / 186624.0 + _ZETA2 / 27.0)
            + (19.0 * _ZETA3) / 1728.0
        )
        + (2428169183.0 * _ZETA3) / 87091200.0
        - (1439.0 * _PI * _PI * _ZETA3) / 1296.0
        + lmM * (
            21084715.0 / 746496.0
            + (35.0 * _ZETA2) / 9.0
            + (35.0 * _LOG2 * _ZETA2) / 27.0
            - (65.0 * _ZETA3) / 108.0
            + (3022001.0 * _ZETA3) / 165888.0
        )
        + nlq * (
            -1773073.0 / 746496.0
            - (9115.0 * lmM * lmM) / 10368.0
            - (107.0 * lmM * lmM * lmM) / 1728.0
            - (967.0 * _PI * _PI) / 1944.0
            + (697709.0 * _PI**4) / 14929920.0
            - (49.0 * _ZETA2) / 108.0
            - (11.0 * _PI * _PI * _LOG2) / 243.0
            + (1709.0 * _PI * _PI * _LOG2 * _LOG2) / 124416.0
            - (173.0 * _LOG2**4) / 124416.0
            - (173.0 * _A4) / 5184.0
            + lmM * (
                -1140191.0 / 373248.0
                - (47.0 * _ZETA2) / 54.0
                - (2.0 * _LOG2 * _ZETA2) / 27.0
                - (7.0 * _ZETA3) / 27.0
                - (110779.0 * _ZETA3) / 82944.0
            )
            - (4756441.0 * _ZETA3) / 995328.0
            - (115.0 * _ZETA5) / 576.0
        )
        + (40596749.0 * _ZETA5) / 1451520.0
    )
    total = 0.0
    for i in range(loops):
        total += sum_terms[i]
    return total


def DecAsDownOS(alphas: float, massth: float, muth: float, nf: int, loops: int) -> float:
    if not 1 <= loops <= 5:
        raise ValueError(f"DecAsDownOS: loops must be 1..5, got {loops}")
    api = alphas / _PI
    return alphas * _fas5to6os(api, massth, muth, float(nf), loops)


def DecAsUpOS(alphas: float, massth: float, muth: float, nf: int, loops: int) -> float:
    if not 1 <= loops <= 5:
        raise ValueError(f"DecAsUpOS: loops must be 1..5, got {loops}")
    api = alphas / _PI
    return alphas * _fas6to5os(api, massth, muth, float(nf), loops)


# ---------------------------------------------------------------------------
# Helpers: αs/π as a function of L = log(μ²/Λ²)   [the setasL series]
# ---------------------------------------------------------------------------

def _api_from_L(L: float, b0: float, b1: float, b2: float,
                b3: float, b4: float, loops: int) -> float:
    """
    Evaluate the loop-expansion of αs/π = f(L) where L = log(μ²/Λ²).

    These are the `as0`..`as4` expressions from the Mathematica setasL
    substitution list, translated term by term.
    """
    logL = math.log(L)
    # 1-loop
    result = 1.0 / (b0 * L)
    if loops == 1:
        return result
    # 2-loop
    result -= b1 * logL / (b0**3 * L**2)
    if loops == 2:
        return result
    # 3-loop
    result += (b2 / b0**4 + b1**2 * (logL**2 - logL - 1) / b0**5) / L**3
    if loops == 3:
        return result
    # 4-loop
    result += (
        b3 / (2 * b0**5)
        - 3 * b1 * b2 * logL / b0**6
        + b1**3 * (-1 / 2 + 2 * logL + 5 * logL**2 / 2 - logL**3) / b0**7
    ) / L**4
    if loops == 4:
        return result
    # 5-loop
    result += (
        b4 / (3 * b0**6)
        + 5 * b2**2 / (3 * b0**7)
        - b1 * b3 * (2 * logL + 1 / 6) / b0**7
        + 3 * b1**2 * b2 * (2 * logL**2 - logL - 1) / b0**8
        + b1**4 * (
            logL**4 - 13 / 3 * logL**3 - 3 / 2 * logL**2 + 4 * logL + 7 / 6
        ) / b0**9
    ) / L**5
    return result


# ---------------------------------------------------------------------------
# LamExpl
# ---------------------------------------------------------------------------

def LamExpl(alphas: float, mu: float, nf: int, loops: int) -> float:
    """
    Compute Λ^(nf) from αs(μ) using the explicit analytical formula.

    Implements the `LamExpl` function from RunDec.m (v3.1), valid for
    1 ≤ loops ≤ 5.

    The formula expresses log(μ²/Λ²) as a power series in αs/π:

        log(μ²/Λ²) = 1/(as·b0) + b1·log(as·b0)/b0²
                    + as·(b1²−b0·b2)/b0³ + …

    and then Λ = μ · exp(−½ · log(μ²/Λ²)), truncated at the given loop order.

    Parameters
    ----------
    alphas : float
        αs(μ) in the MS-bar scheme.
    mu : float
        Renormalization scale (GeV).
    nf : int
        Number of active flavours.
    loops : int
        Loop order, 1 ≤ loops ≤ 5.

    Returns
    -------
    float
        Λ^(nf) in GeV.
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"LamExpl: loops must be 1..5, got {loops}")

    api = alphas / math.pi
    b0 = beta0(nf)
    b1 = beta1(nf)
    b2 = beta2(nf)
    b3 = beta3(nf)
    b4 = beta4(nf)

    # log(Λ/μ) = −½ · log(μ²/Λ²), accumulated term by term.
    # Each term is labelled with the loop order at which it first appears.
    lamomu = -1.0 / (2.0 * api * b0)  # 1-loop

    if loops >= 2:
        # The log(as) + log(b0) = log(as·b0) term from logmu2lam2
        lamomu += -b1 / (2.0 * b0**2) * math.log(api * b0)

    if loops >= 3:
        lamomu += api * (b1**2 - b0 * b2) / (2.0 * b0**3)

    if loops >= 4:
        lamomu += -api**2 * (b1**3 - 2 * b0 * b1 * b2 + b0**2 * b3) / (4.0 * b0**4)

    if loops >= 5:
        lamomu += api**3 * (
            b1**4
            - 3 * b0 * b1**2 * b2
            + b0**2 * b2**2
            + 2 * b0**2 * b1 * b3
            - b0**3 * b4
        ) / (6.0 * b0**5)

    return mu * math.exp(lamomu)


# ---------------------------------------------------------------------------
# LamImpl
# ---------------------------------------------------------------------------

def LamImpl(alphas: float, mu: float, nf: int, loops: int) -> float:
    """
    Compute Λ^(nf) from αs(μ) by numerically inverting AlphasLam.

    Implements the `LamImpl` function from RunDec.m (v3.1).  Uses Newton's
    method (scipy.optimize.newton) with LamExpl as the initial guess, which
    mirrors Mathematica's FindRoot behaviour.

    Note: not JAX-differentiable.  Use LamExpl when gradients are needed.

    Parameters
    ----------
    alphas : float
        αs(μ) in the MS-bar scheme.
    mu : float
        Renormalization scale (GeV).
    nf : int
        Number of active flavours.
    loops : int
        Loop order, 1 ≤ loops ≤ 5.

    Returns
    -------
    float
        Λ^(nf) in GeV.
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"LamImpl: loops must be 1..5, got {loops}")

    b0 = beta0(nf)
    b1 = beta1(nf)
    b2 = beta2(nf)
    b3 = beta3(nf)
    b4 = beta4(nf)
    target_api = alphas / math.pi
    lam0 = LamExpl(alphas, mu, nf, loops)

    def equation(lam: float) -> float:
        L = math.log(mu**2 / lam**2)
        return _api_from_L(L, b0, b1, b2, b3, b4, loops) - target_api

    return float(newton(equation, lam0, tol=1e-14, maxiter=50))


# ---------------------------------------------------------------------------
# AlphasLam
# ---------------------------------------------------------------------------

def AlphasLam(lam: float, mu: float, nf: int, loops: int) -> float:
    """
    Compute αs(μ) from Λ^(nf) using the explicit loop expansion.

    Implements the `AlphasLam` function from RunDec.m (v3.1), valid for
    1 ≤ loops ≤ 5.

    Parameters
    ----------
    lam : float
        Λ^(nf) in GeV.
    mu : float
        Renormalization scale (GeV).
    nf : int
        Number of active flavours.
    loops : int
        Loop order, 1 ≤ loops ≤ 5.

    Returns
    -------
    float
        αs(μ).
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"AlphasLam: loops must be 1..5, got {loops}")

    ratio = mu / lam
    if ratio < _RMULAM:
        import warnings
        warnings.warn(
            f"AlphasLam: the ratio μ/Λ = {ratio:.4f} is very small "
            f"(< {_RMULAM}).  Result may be unreliable.",
            stacklevel=2,
        )

    b0 = beta0(nf)
    b1 = beta1(nf)
    b2 = beta2(nf)
    b3 = beta3(nf)
    b4 = beta4(nf)
    L = math.log(mu**2 / lam**2)
    return math.pi * _api_from_L(L, b0, b1, b2, b3, b4, loops)


# ---------------------------------------------------------------------------
# AlphasExact
# ---------------------------------------------------------------------------

def AlphasExact(
    alphas0: float,
    mu0: float,
    mu: float,
    nf: int,
    loops: int,
) -> float:
    """
    Compute αs(μ) by integrating the RGE numerically with diffrax.

    Implements the `AlphasExact` function from RunDec.m (v3.1), valid for
    1 ≤ loops ≤ 5.

    The ODE solved is

        d(api) / d(log μ²) = −api² (b0 + b1·api + b2·api² + …)

    where api = αs/π.  Integration is performed from log(μ0²) to log(μ²)
    using Dormand–Prince (Dopri5) with adaptive step-size control.

    This function is JAX-JIT-able when `nf` and `loops` are declared as
    static arguments::

        from functools import partial
        import jax
        exact_jit = jax.jit(partial(AlphasExact, nf=5, loops=5))

    Parameters
    ----------
    alphas0 : float
        αs(μ0) initial condition.
    mu0 : float
        Initial renormalization scale (GeV).
    mu : float
        Target renormalization scale (GeV).
    nf : int
        Number of active flavours.
    loops : int
        Loop order, 1 ≤ loops ≤ 5.

    Returns
    -------
    float
        αs(μ).
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"AlphasExact: loops must be 1..5, got {loops}")
    if mu0 == mu:
        return alphas0

    # Check μ/Λ at target scale
    lam = LamExpl(alphas0, mu0, nf, min(loops, 5))
    if mu / lam < _RMULAM:
        import warnings
        warnings.warn(
            f"AlphasExact: the ratio μ/Λ = {mu / lam:.4f} is very small "
            f"(< {_RMULAM}).  Result may be unreliable.",
            stacklevel=2,
        )

    b0 = beta0(nf)
    b1 = beta1(nf)
    b2 = beta2(nf)
    b3 = beta3(nf)
    b4 = beta4(nf)

    # RHS of  d(api)/d(t)  with  t = log(μ²)
    # Selecting terms by loop order here (Python int, not traced) is safe
    # under jax.jit with static nf/loops.
    if loops == 1:
        def _rhs(t, api, args):
            return -api**2 * b0
    elif loops == 2:
        def _rhs(t, api, args):
            return -api**2 * (b0 + b1 * api)
    elif loops == 3:
        def _rhs(t, api, args):
            return -api**2 * (b0 + b1 * api + b2 * api**2)
    elif loops == 4:
        def _rhs(t, api, args):
            return -api**2 * (b0 + b1 * api + b2 * api**2 + b3 * api**3)
    else:  # loops == 5
        def _rhs(t, api, args):
            return -api**2 * (b0 + b1 * api + b2 * api**2 + b3 * api**3 + b4 * api**4)

    y0 = jnp.asarray(alphas0 / math.pi)
    t0 = math.log(mu0**2)
    t1 = math.log(mu**2)
    dt0 = (t1 - t0) / 100.0  # initial step, sign matches integration direction

    sol = diffeqsolve(
        ODETerm(_rhs),
        Dopri5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        stepsize_controller=PIDController(rtol=1e-10, atol=1e-12),
        max_steps=10_000,
    )
    return float(sol.ys[-1] * math.pi)


# ---------------------------------------------------------------------------
# Threshold running for αs (low->high and high->low)
# ---------------------------------------------------------------------------

def _check_consecutive_nfs(nfs: Sequence[int], direction: int) -> None:
    for i in range(1, len(nfs)):
        if nfs[i] - nfs[i - 1] != direction:
            raise ValueError("There is a gap in the number of flavours.")


def AlL2AlH(
    alphas_low: float,
    mu1: float,
    decpar: Iterable[object],
    mu2: float,
    loops: int,
) -> float:
    """
    Run αs from low to high scale with decoupling at thresholds (OS scheme).

    Parameters match CRunDec::AlL2AlH:
      alphas_low = αs(μ1), decpar = thresholds, mu2 = target scale.
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"AlL2AlH: loops must be 1..5, got {loops}")

    entries = _normalize_decpar(decpar)
    if not entries:
        raise ValueError("decpar must contain at least one threshold")

    entries = sorted(entries, key=lambda d: d.nf)
    _check_consecutive_nfs([d.nf for d in entries], direction=1)

    asini = alphas_low
    muini = mu1
    for dec in entries:
        as_at_muth = AlphasExact(asini, muini, dec.muth, nf=dec.nf - 1, loops=loops)
        asini = DecAsUpOS(as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        muini = dec.muth

    return AlphasExact(asini, muini, mu2, nf=entries[-1].nf, loops=loops)


def AlH2AlL(
    alphas_high: float,
    mu1: float,
    decpar: Iterable[object],
    mu2: float,
    loops: int,
) -> float:
    """
    Run αs from high to low scale with decoupling at thresholds (OS scheme).

    Parameters match CRunDec::AlH2AlL:
      alphas_high = αs(μ1), decpar = thresholds, mu2 = target scale.
    """
    if not 1 <= loops <= 5:
        raise ValueError(f"AlH2AlL: loops must be 1..5, got {loops}")

    entries = _normalize_decpar(decpar)
    if not entries:
        raise ValueError("decpar must contain at least one threshold")

    entries = sorted(entries, key=lambda d: d.nf, reverse=True)
    _check_consecutive_nfs([d.nf for d in entries], direction=-1)

    asini = alphas_high
    muini = mu1
    for dec in entries:
        as_at_muth = AlphasExact(asini, muini, dec.muth, nf=dec.nf, loops=loops)
        asini = DecAsDownOS(as_at_muth, dec.Mth, dec.muth, nf=dec.nf - 1, loops=loops)
        muini = dec.muth

    return AlphasExact(asini, muini, mu2, nf=entries[-1].nf - 1, loops=loops)



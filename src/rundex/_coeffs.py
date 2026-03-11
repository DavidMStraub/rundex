"""
Beta-function and anomalous-dimension coefficients for QCD.

All functions take `nf` (number of active flavours) as a plain Python int
and return a plain Python float.  No JAX arrays are produced here so these
can be called freely at module-import time.

Conventions
-----------
The QCD β-function is defined as

    μ² d(αs)/d(μ²) = β(αs)
    β(αs) = −αs² (b0 + b1 αs + b2 αs² + …)      (αs ≡ α_s/π)

The quark-mass anomalous dimension is

    μ² d(ln m)/d(μ²) = γ_m(αs)
    γ_m(αs) = −αs (g0 + g1 αs + g2 αs² + …)

Colour factors are fixed to the QCD values  CA=3, CF=4/3, TR=1/2.

References
----------
β-function through 5 loops: arXiv:1606.08659 (Baikov, Chetyrkin, Kühn)
γ_m through 5 loops:        arXiv:1402.6611  (Luthe, Maier, Marquard, Schröder)
"""

import math


# ---------------------------------------------------------------------------
# Transcendental constants used in the coefficients
# ---------------------------------------------------------------------------

_z2 = math.pi**2 / 6          # ζ(2) = π²/6
_z3 = 1.2020569031595942854    # ζ(3) — Apéry's constant
_z4 = math.pi**4 / 90         # ζ(4) = π⁴/90
_z5 = 1.0369277551433699341    # ζ(5)
_z6 = math.pi**6 / 945        # ζ(6) = π⁶/945
_z7 = 1.0083492773819228268    # ζ(7)

_log2 = math.log(2)

# Li_4(1/2) — needed for the constant B4
# B4 = 16·Li₄(1/2) + (2/3)·ln²(2)² − (2/3)π²·ln²(2) − (13/180)π⁴
_Li4_half = 0.5174790616738993145  # 20 significant digits
_Li5_half = 0.5084005792422687014

_B4 = (
    16 * _Li4_half
    + (2 / 3) * _log2**4
    - (2 / 3) * math.pi**2 * _log2**2
    - (13 / 180) * math.pi**4
)

# ---------------------------------------------------------------------------
# β-function coefficients  b_i  (the usual normalisations, scheme: MS-bar)
# ---------------------------------------------------------------------------

def beta0(nf: int) -> float:
    """1-loop β-function coefficient."""
    return 11 / 4 - nf / 6


def beta1(nf: int) -> float:
    """2-loop β-function coefficient."""
    return 51 / 8 - 19 * nf / 24


def beta2(nf: int) -> float:
    """3-loop β-function coefficient."""
    return 2857 / 128 - 5033 * nf / 1152 + 325 * nf**2 / 3456


def beta3(nf: int) -> float:
    """4-loop β-function coefficient (arXiv:hep-ph/9701390)."""
    return (
        149753 / 1536
        - 1078361 * nf / 41472
        + 50065 * nf**2 / 41472
        + 1093 * nf**3 / 186624
        + 891 * _z3 / 64
        - 1627 * nf * _z3 / 1728
        + 809 * nf**2 * _z3 / 2592
    )


def beta4(nf: int) -> float:
    """5-loop β-function coefficient (arXiv:1606.08659)."""
    nf2, nf3, nf4 = nf**2, nf**3, nf**4
    return (
        8157455 / 16384
        - 336460813 * nf / 1990656
        + 25960913 * nf2 / 1990656
        - 630559 * nf3 / 5971968
        + 1205 * nf4 / 2985984
        # ζ(3) terms
        + 621885 * _z3 / 2048
        - 1202791 * nf * _z3 / 20736
        + 698531 * nf2 * _z3 / 82944
        - 24361 * nf3 * _z3 / 124416
        - 19 * nf4 * _z3 / 10368
        # ζ(4) terms
        - 88209 * _z4 / 2048
        + 33935 * nf * _z4 / 6144
        - 5263 * nf2 * _z4 / 4608
        + 809 * nf3 * _z4 / 13824
        # ζ(5) terms
        - 144045 * _z5 / 512
        + 1358995 * nf * _z5 / 27648
        - 5965 * nf2 * _z5 / 1296
        + 115 * nf3 * _z5 / 2304
    )


# ---------------------------------------------------------------------------
# γ_m anomalous-dimension coefficients  g_i
# ---------------------------------------------------------------------------

def gamma0(nf: int) -> float:  # noqa: ARG001
    """1-loop mass anomalous dimension coefficient."""
    return 1.0


def gamma1(nf: int) -> float:
    """2-loop mass anomalous dimension coefficient."""
    return (101 / 2 - 5 * nf / 3) / 12


def gamma2(nf: int) -> float:
    """3-loop mass anomalous dimension coefficient."""
    return (3747 / 4 - nf * (1108 / 9 + 70 * nf / 27 + 80 * _z3) / 2) / 48


def gamma3(nf: int) -> float:
    """4-loop mass anomalous dimension coefficient (arXiv:hep-ph/9703278)."""
    nf2, nf3 = nf**2, nf**3
    pi4 = math.pi**4
    return (
        4603055 / 41472
        - 91723 * nf / 6912
        + 2621 * nf2 / 31104
        - 83 * nf3 / 15552
        + 11 * nf * pi4 / 288
        - nf2 * pi4 / 432
        + 530 * _z3 / 27
        - 2137 * nf * _z3 / 144
        + 25 * nf2 * _z3 / 72
        + nf3 * _z3 / 108
        - 275 * _z5 / 8
        + 575 * nf * _z5 / 72
    )


def gamma4(nf: int) -> float:
    """5-loop mass anomalous dimension coefficient (arXiv:1402.6611)."""
    nf2, nf3, nf4 = nf**2, nf**3, nf**4
    z3sq = _z3**2
    return (1 / 4) ** 5 * (
        99512327 / 162
        + 46402466 / 243 * _z3
        + 96800 * z3sq
        - 698126 / 9 * _z4
        - 231757160 / 243 * _z5
        + 242000 * _z6
        + 412720 * _z7
        + nf * (
            -150736283 / 1458
            - 12538016 / 81 * _z3
            - 75680 / 9 * z3sq
            + 2038742 / 27 * _z4
            + 49876180 / 243 * _z5
            - 638000 / 9 * _z6
            - 1820000 / 27 * _z7
        )
        + nf2 * (
            1320742 / 729
            + 2010824 / 243 * _z3
            + 46400 / 27 * z3sq
            - 166300 / 27 * _z4
            - 264040 / 81 * _z5
            + 92000 / 27 * _z6
        )
        + nf3 * (91865 / 1458 + 12848 / 81 * _z3 + 448 / 9 * _z4 - 5120 / 27 * _z5)
        + nf4 * (-260 / 243 - 320 / 243 * _z3 + 64 / 27 * _z4)
    )

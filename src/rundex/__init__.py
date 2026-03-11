"""
RunDec — JAX port of the Mathematica package RunDec v3.1
(running and decoupling of the strong coupling and quark masses)

Original authors: K.G. Chetyrkin, J.H. Kühn, M. Steinhauser,
                  F. Herren, M. Steinhauser
References: hep-ph/0004189, arXiv:1703.03751
"""

from ._alphas import (
    AlH2AlL,
    AlL2AlH,
    AlphasExact,
    AlphasLam,
    Decoupling,
    LamExpl,
    LamImpl,
)
from ._masses import (
    DecMqDownOS,
    DecMqUpOS,
    mH2mL,
    mL2mH,
    mMS2mMS,
    mMS2mOS,
    mMS2m1S,
    mOS2mMS,
)

__all__ = [
    "LamExpl",
    "LamImpl",
    "AlphasLam",
    "AlphasExact",
    "Decoupling",
    "AlL2AlH",
    "AlH2AlL",
    "mMS2mMS",
    "mMS2mOS",
    "mMS2m1S",
    "mOS2mMS",
    "DecMqUpOS",
    "DecMqDownOS",
    "mL2mH",
    "mH2mL",
]
__version__ = "3.1.0"

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

__all__ = [
    "LamExpl",
    "LamImpl",
    "AlphasLam",
    "AlphasExact",
    "Decoupling",
    "AlL2AlH",
    "AlH2AlL",
]
__version__ = "3.1.0"

# rundex

JAX port of [RunDec](https://www.ttp.kit.edu/Progdata/ttp16/ttp16-014/) v3.1 — running and decoupling of the strong coupling αs and quark masses in QCD.

Original Mathematica package by K.G. Chetyrkin, J.H. Kühn, M. Steinhauser, and F. Herren.  
References: [hep-ph/0004189](https://arxiv.org/abs/hep-ph/0004189), [arXiv:1703.03751](https://arxiv.org/abs/1703.03751).

## Installation


```bash
pip install rundex
```

Requires Python ≥ 3.10, JAX ≥ 0.4, and [diffrax](https://github.com/patrick-kidger/diffrax).


## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)  # required for physics-level precision

from rundex import AlphasExact, AlphasLam, LamExpl, LamImpl

# Λ_MS^(5) from αs(MZ)
lam = LamExpl(0.118, 91.18, nf=5, loops=5)   # → 0.2086 GeV

# αs(μ) from Λ (analytic expansion)
als = AlphasLam(lam, mu=1000.0, nf=5, loops=5)

# αs(μ) by numerically integrating the RGE
als = AlphasExact(0.118, mu0=91.18, mu=1000.0, nf=5, loops=5)

# Optional: compile once for fixed (nf, loops) to avoid retracing
from rundex import compiled_AlphasExact
alphas_exact_5l_nf5 = compiled_AlphasExact(nf=5, loops=5)
als_fast = alphas_exact_5l_nf5(0.118, 91.18, 1000.0)
```

> **Note on 64-bit precision**  
> JAX defaults to 32-bit floats. Always call
> `jax.config.update("jax_enable_x64", True)` **before** importing or calling
> any `rundex` function. The package itself does not set this flag so it does
> not interfere with the rest of your JAX code.

## Development

```bash
git clone ...
pip install -e ".[dev]"
pytest
```

Numerical tests are cross-validated against [rundec-python](https://github.com/apaasch/rundec-python) (CRunDec C++ wrapper), which is a test-only dependency not required by rundex itself.

## Implemented RunDec functions (subset)

The following RunDec functions are currently ported to `rundex`:

- `LamExpl`
- `LamImpl`
- `AlphasLam`
- `AlphasExact`
- `AlL2AlH`
- `AlH2AlL`
- `mMS2mMS`
- `mMS2mOS`
- `mMS2m1S`
- `mOS2mMS`
- `mOS2mSI`
- `DecMqUpOS`
- `DecMqDownOS`
- `mL2mH`
- `mH2mL`

Not all RunDec functionality is ported yet.

## Compiled helpers (available for all ported functions)

Each ported function above has a compiled helper named `compiled_<FunctionName>`.
Use these when you call the same function many times with fixed discrete
settings (for example fixed `nf`, `loops`, and fixed threshold/light-quark data).

Examples:

- `compiled_LamExpl(nf, loops)`
- `compiled_LamImpl(nf, loops)`
- `compiled_AlphasLam(nf, loops)`
- `compiled_AlphasExact(nf, loops)`
- `compiled_AlL2AlH(decpar, loops)`
- `compiled_AlH2AlL(decpar, loops)`
- `compiled_mMS2mMS(nf, loops)`
- `compiled_mMS2mOS(nf, loops, mq=None, fdelm=1.0)`
- `compiled_mMS2m1S(nl, loops, mq=None, fdelm=1.0)`
- `compiled_mOS2mMS(nf, loops, mq=None, fdelm=1.0)`
- `compiled_mOS2mSI(nf, loops, mq=None, fdelm=1.0)`
- `compiled_DecMqUpOS(nf, loops)`
- `compiled_DecMqDownOS(nf, loops)`
- `compiled_mL2mH(decpar, loops)`
- `compiled_mH2mL(decpar, loops)`

When to use which:

- Use regular functions for one-off calls or frequently changing setup parameters.
- Use compiled helpers for repeated calls with fixed setup parameters.

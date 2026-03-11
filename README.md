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
```

> **Note on 64-bit precision**  
> JAX defaults to 32-bit floats. Always call
> `jax.config.update("jax_enable_x64", True)` **before** importing or calling
> any `rundex` function. The package itself does not set this flag so it does
> not interfere with the rest of your JAX code.


## Conventions

- Scales (μ, Λ) in GeV.
- `nf` is a plain Python `int` and is never JAX-traced.
- `loops` is a plain Python `int`.
- The β-function is  
  $\mu^2 \frac{d\alpha_s}{d\mu^2} = -\alpha_s^2 (b_0 + b_1 \alpha_s + \cdots)$  
  with $\alpha_s \equiv \alpha_s$ (not $\alpha_s/\pi$).

## Development

```bash
git clone ...
pip install -e ".[dev]"
pytest
```

Numerical tests are cross-validated against [rundec-python](https://github.com/apaasch/rundec-python) (CRunDec C++ wrapper), which is a test-only dependency not required by rundex itself.

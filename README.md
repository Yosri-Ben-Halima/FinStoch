# FinStoch

[![PyPI Latest Release](https://img.shields.io/pypi/v/FinStoch.svg)](https://pypi.org/project/FinStoch/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/FinStoch.svg?label=PyPI%20downloads)](https://pypi.org/project/FinStoch/)
[![License - MIT](https://img.shields.io/pypi/l/FinStoch.svg)](https://github.com/Yosri-Ben-Halima/FinStoch/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/FinStoch.svg)](https://pypi.org/project/FinStoch/)
[![CI](https://github.com/Yosri-Ben-Halima/FinStoch/actions/workflows/ci.yml/badge.svg)](https://github.com/Yosri-Ben-Halima/FinStoch/actions/workflows/ci.yml)
[![Typing](https://img.shields.io/pypi/types/FinStoch)](https://pypi.org/project/FinStoch/)

## What is it?

**FinStoch** is a Python library for simulating stochastic processes commonly used in quantitative finance. It provides clean, consistent interfaces for Monte Carlo path simulation using Euler-Maruyama discretization, with built-in analytics, seed control for reproducibility, and pandas integration.

- **Source code:** <https://github.com/Yosri-Ben-Halima/FinStoch>
- **Bug reports:** <https://github.com/Yosri-Ben-Halima/FinStoch/issues>
- **PyPI:** <https://pypi.org/project/FinStoch/>

## Table of Contents

- [Main Features](#main-features)
- [Supported Processes](#supported-processes)
- [Where to Get It](#where-to-get-it)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Analytics](#analytics)
- [Development](#development)
- [License](#license)

## Main Features

- **Six stochastic process models** covering equity prices, interest rates, and stochastic volatility
- **Reproducible simulations** via seed control on all processes
- **Flexible time grids** with configurable granularity (daily, hourly, minute-level) and business day support
- **Built-in analytics** including VaR, CVaR, max drawdown, confidence bands, and summary statistics
- **pandas integration** with `to_dataframe()` for seamless downstream analysis
- **Consistent API** across all models: every process exposes `simulate()`, `plot()`, and the full analytics suite

## Supported Processes

| Model | Class | SDE |
| --- | --- | --- |
| Geometric Brownian Motion | `GeometricBrownianMotion` | $dS = \mu S\, dt + \sigma S\, dW$ |
| Merton Jump Diffusion | `MertonJumpDiffusion` | $dS = (\mu - \lambda k) S\, dt + \sigma S\, dW + JS\, dN$ |
| Ornstein-Uhlenbeck | `OrnsteinUhlenbeck` | $dS = \theta(\mu - S)\, dt + \sigma\, dW$ |
| Cox-Ingersoll-Ross | `CoxIngersollRoss` | $dS = \theta(\mu - S)\, dt + \sigma\sqrt{S}\, dW$ |
| Constant Elasticity of Variance | `ConstantElasticityOfVariance` | $dS = \mu S\, dt + \sigma S^\gamma\, dW$ |
| Heston Stochastic Volatility | `HestonModel` | $dS = \mu S\, dt + \sqrt{v} S\, dW_S$, $dv = \kappa(\theta - v)\, dt + \sigma\sqrt{v}\, dW_v$ |

All processes are discretized using the Euler-Maruyama scheme and return NumPy arrays of shape `(num_paths, num_steps)`. The Heston model returns a tuple `(S, v)` of price and variance paths.

## Where to Get It

```bash
# PyPI
pip install FinStoch
```

## Dependencies

| Package | Minimum Version | Purpose |
| --- | --- | --- |
| [NumPy](https://numpy.org) | 1.23 | Array operations and random number generation |
| [pandas](https://pandas.pydata.org) | 2.0 | Time grid generation and DataFrame conversion |
| [matplotlib](https://matplotlib.org) | 3.7 | Path visualization |
| [SciPy](https://scipy.org) | 1.9 | Statistical functions for analytics |
| [python-dateutil](https://dateutil.readthedocs.io) | 2.9 | Date range duration calculation |

## Quick Start

### Simulate and plot

```python
from FinStoch import GeometricBrownianMotion

gbm = GeometricBrownianMotion(
    S0=100, mu=0.05, sigma=0.2,
    num_paths=10,
    start_date='2023-09-01',
    end_date='2024-09-01',
    granularity='D',
)

# Reproducible simulation
paths = gbm.simulate(seed=42)
gbm.plot(paths=paths, title='GBM Simulation', ylabel='Price')
```

### Convert to DataFrame

```python
df = gbm.to_dataframe(paths)
# DataFrame with DatetimeIndex columns, one row per path
```

### Heston model (stochastic volatility)

```python
from FinStoch import HestonModel

heston = HestonModel(
    S0=100, v0=0.04, mu=0.05, sigma=0.3,
    theta=0.04, kappa=2.0, rho=-0.7,
    num_paths=10,
    start_date='2023-09-01',
    end_date='2024-09-01',
    granularity='D',
)

prices, variance = heston.simulate(seed=42)

# Convert either component to DataFrame
df_prices = heston.to_dataframe((prices, variance), variance=False)
df_var = heston.to_dataframe((prices, variance), variance=True)
```

## Analytics

All processes inherit a suite of analytics methods from the base class:

```python
paths = gbm.simulate(seed=42)

# Descriptive statistics at each time step
stats = gbm.summary_statistics(paths)  # dict: mean, std, skew, kurtosis, min, max

# Central tendency
mean_path = gbm.expected_path(paths)     # mean across paths
median = gbm.median_path(paths)          # median across paths

# Uncertainty
lower, upper = gbm.confidence_bands(paths, level=0.95)

# Risk measures (computed at terminal time step)
gbm.var(paths, alpha=0.05)    # Value at Risk
gbm.cvar(paths, alpha=0.05)   # Conditional VaR (Expected Shortfall)

# Drawdown analysis
drawdowns = gbm.max_drawdown(paths)  # max peak-to-trough per path

# Distribution visualization
gbm.terminal_distribution(paths, bins=50)  # histogram + fitted normal
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m unittest discover -s tests -p "*_test.py"

# Format
ruff format

# Lint
flake8 --max-line-length 127

# Type check
mypy . --exclude venv --exclude build --ignore-missing-imports
```

## License

[MIT](LICENSE)

<h1 align="center">
<img src="https://raw.githubusercontent.com/Yosri-Ben-Halima/FinStoch/main/logos/finstoch-final-horizontal.svg" width="800">
</h1><br>

[![PyPI Latest Release](https://img.shields.io/pypi/v/FinStoch.svg)](https://pypi.org/project/FinStoch/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/FinStoch.svg?label=PyPI%20downloads)](https://pypi.org/project/FinStoch/)
[![License - MIT](https://img.shields.io/pypi/l/FinStoch.svg)](https://github.com/Yosri-Ben-Halima/FinStoch/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/FinStoch.svg)](https://pypi.org/project/FinStoch/)
[![CI](https://github.com/Yosri-Ben-Halima/FinStoch/actions/workflows/ci.yml/badge.svg)](https://github.com/Yosri-Ben-Halima/FinStoch/actions/workflows/ci.yml)
[![Typing](https://img.shields.io/pypi/types/FinStoch)](https://pypi.org/project/FinStoch/)

## What is it?

**FinStoch** is a Python library for simulating stochastic processes commonly used in quantitative finance. It provides clean, consistent interfaces for Monte Carlo path simulation with various methods available, with built-in analytics, seed control for reproducibility, and pandas integration.

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
- [Calibration](#calibration)
- [Development](#development)
- [License](#license)

## Main Features

- **Nine parametric stochastic process models** covering equity prices, interest rates, jump diffusions, and stochastic volatility
- **Bootstrap Monte Carlo** simulation from historical data (i.i.d. and block bootstrap)
- **Milstein scheme** for higher-order discretization accuracy via `method="milstein"`
- **Exact simulation** via closed-form transition densities where available via `method="exact"`
- **Reproducible simulations** via seed control on all processes
- **Flexible time grids** with configurable granularity (daily, hourly, minute-level) and business day support
- **Built-in analytics** including VaR, CVaR, max drawdown, confidence bands, and summary statistics
- **pandas integration** with `to_dataframe()` for seamless downstream analysis
- **Parameter calibration** from observed data via `calibrate()` using literature-backed estimation methods (MLE, CLS, EM, GMM)
- **Consistent API** across all models: every process exposes `simulate()`, `plot()`, `calibrate()`, and the full analytics suite

## Supported Processes

| Model | Class | Description |
| --- | --- | --- |
| Geometric Brownian Motion | `GeometricBrownianMotion` | Log-normal asset price dynamics with constant drift and volatility |
| Merton Jump Diffusion | `MertonJumpDiffusion` | GBM extended with Poisson-driven jumps for sudden price shocks |
| Ornstein-Uhlenbeck | `OrnsteinUhlenbeck` | Mean-reverting process with constant volatility |
| Cox-Ingersoll-Ross | `CoxIngersollRoss` | Mean-reverting, non-negative process with square-root volatility |
| Constant Elasticity of Variance | `ConstantElasticityOfVariance` | GBM generalization where volatility scales as a power of price |
| Heston Stochastic Volatility | `HestonModel` | Asset price with mean-reverting stochastic variance and correlation |
| Vasicek | `VasicekModel` | Mean-reverting interest rate model (allows negative rates) |
| Bates | `BatesModel` | Heston stochastic volatility combined with Merton-style jumps |
| Variance Gamma | `VarianceGammaProcess` | Pure-jump process via time-changed Brownian motion with heavier tails |

All parametric processes return NumPy arrays of shape `(num_paths, num_steps)`. Heston and Bates return a tuple `(S, v)` of price and variance paths. Discretization uses Euler-Maruyama by default; pass `method="milstein"` for higher-order accuracy or `method="exact"` for exact transition density sampling (available for GBM, Merton, OU, Vasicek, CIR, and Variance Gamma).

### Non-parametric

| Method | Class | Description |
| --- | --- | --- |
| Bootstrap Monte Carlo | `BootstrapMonteCarlo` | Resamples historical log returns (i.i.d. or block bootstrap) |

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

### End-to-end workflow

```python
import numpy as np
from FinStoch import GeometricBrownianMotion

# 1. Calibrate from historical data
historical_prices = np.array([...])  # 1D array of daily prices
params = GeometricBrownianMotion.calibrate(historical_prices, dt=1/252)

# 2. Initialize the model with calibrated parameters
gbm = GeometricBrownianMotion(
    S0=historical_prices[-1], **params,
    num_paths=1000,
    start_date='2024-01-01',
    end_date='2025-01-01',
    granularity='D',
)

# 3. Simulate future paths
paths = gbm.simulate(seed=42)

# 4. Analyze
gbm.var(paths, alpha=0.05)                # Value at Risk
gbm.cvar(paths, alpha=0.05)               # Expected Shortfall
lower, upper = gbm.confidence_bands(paths) # 95% confidence bands
drawdowns = gbm.max_drawdown(paths)        # max peak-to-trough per path
df = gbm.to_dataframe(paths)               # pandas DataFrame

# 5. Visualize
gbm.plot(paths=paths, title='GBM Forecast', ylabel='Price')
gbm.terminal_distribution(paths)
```

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

### Milstein scheme

Use the higher-order Milstein discretization for improved accuracy:

```python
# Euler-Maruyama (default)
paths_euler = gbm.simulate(seed=42, method='euler')

# Milstein scheme
paths_milstein = gbm.simulate(seed=42, method='milstein')
```

The Milstein scheme is available on all Euler-Maruyama-based processes. For OU and Vasicek (constant diffusion), it is identical to Euler. For Variance Gamma (time-changed Brownian motion), it raises `ValueError`.

### Exact simulation

Use exact transition densities for processes with closed-form solutions:

```python
from FinStoch import OrnsteinUhlenbeck

ou = OrnsteinUhlenbeck(
    S0=100, mu=50, sigma=5, theta=0.5,
    num_paths=10,
    start_date='2023-01-01',
    end_date='2024-01-01',
    granularity='D',
)
paths_exact = ou.simulate(seed=42, method='exact')
```

Available for: GBM, Merton (alias for euler), OU, Vasicek, CIR, Variance Gamma. For processes without a closed-form solution (CEV, Heston, Bates, Bootstrap), `method="exact"` falls back to Euler-Maruyama with a warning.

### Bootstrap Monte Carlo

Simulate from historical data without assuming a parametric model:

```python
from FinStoch.bootstrap import BootstrapMonteCarlo
import numpy as np

# Historical daily prices (e.g. from yfinance)
prices = np.array([100, 102, 99, 103, 101, 104, 98, 105, 107, 103])

model = BootstrapMonteCarlo(
    historical_prices=prices,
    num_paths=1000,
    start_date='2024-01-01',
    end_date='2024-06-01',
    granularity='D',
    # S0 defaults to last price; block_size=5 for block bootstrap
)

paths = model.simulate(seed=42)
model.var(paths, alpha=0.05)  # all analytics inherited
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

## Calibration

All parametric processes provide a `calibrate()` class method that estimates model parameters from observed data:

```python
import numpy as np
from FinStoch import GeometricBrownianMotion

# Historical daily prices
prices = np.array([...])  # 1D array

# Estimate parameters (dt=1/252 for daily data)
params = GeometricBrownianMotion.calibrate(prices, dt=1/252)
# {'mu': 0.08, 'sigma': 0.2}

# Use calibrated params to build a simulation
gbm = GeometricBrownianMotion(
    S0=prices[-1], **params,
    num_paths=1000,
    start_date='2024-01-01',
    end_date='2025-01-01',
    granularity='D',
)
```

| Model | Method | Reference |
| --- | --- | --- |
| GBM | Exact MLE on log-returns | Cont & Tankov (2004) |
| Ornstein-Uhlenbeck | AR(1) MLE | Vasicek (1977) |
| Vasicek | AR(1) MLE | Vasicek (1977) |
| Cox-Ingersoll-Ross | Conditional Least Squares | Overbeck & Rydberg (1997) |
| CEV | CKLS Quasi-MLE | Chan, Karolyi, Longstaff, Sanders (1992) |
| Merton Jump Diffusion | EM algorithm | Honore (1998) |
| Heston | Realized variance + CIR | Bollerslev & Zhou (2002) |
| Bates | Two-stage (Heston + jump EM) | Bates (1996) |
| Variance Gamma | Method of Moments | Madan, Carr, Chang (1998) |

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

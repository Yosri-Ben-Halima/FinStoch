"""
The `FinStoch.metrics` module contains classes and methods for calculating various Monte-Carlo metrics.
"""

from .var import ValueAtRisk
from .cvar import ExpectedShortfall
from .returns import ExpectedReturn
from .vol import Volatility
from .entropy import Entropy
from .sharpe import SharpeRatio
from .sortino import SortinoRatio
from .drawdown import MaxDrawdown
from .skewness import Skewness
from .kurtosis import Kurtosis

__all__ = [
    "ValueAtRisk",
    "ExpectedShortfall",
    "Volatility",
    "ExpectedReturn",
    "MaxDrawdown",
    "Skewness",
    "Kurtosis",
    "SortinoRatio",
    "Entropy",
    "SharpeRatio",
]

__version__ = "v1.0.0"

__author__ = "Yosri Ben Halima"

__email__ = "yosri.benhalima@ept.ucar.tn"

__description__ = "A financial model simulation and risk evaluation library."

__url__ = "https://github.com/Yosri-Ben-Halima/FinStoch"

__license__ = "MIT"

__copyright__ = "Copyright (c) 2024 Yosri Ben Halima"

"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

from .gbm import GeometricBrownianMotion
from .merton import MertonJumpDiffusion
from .ou import OrnsteinUhlenbeck
from .cir import CoxIngersollRoss
from .heston import HestonModel
from .cev import ConstantElasricityOfVariance

__all__ = [
    "GeometricBrownianMotion",
    "MertonJumpDiffusion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "HestonModel",
    "ConstantElasricityOfVariance",
]

__version__ = "v1.0.0"

__author__ = "Yosri Ben Halima"

__email__ = "yosri.benhalima@ept.ucar.tn"

__description__ = "A financial model simulation and risk evaluation library."

__url__ = "https://github.com/Yosri-Ben-Halima/FinStoch"

__license__ = "MIT"

__copyright__ = "Copyright (c) 2024 Yosri Ben Halima"

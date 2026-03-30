"""FinStoch — A Python library for simulating stochastic processes in finance."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("FinStoch")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Yosri Ben Halima"
__email__ = "yosri.benhalima@ept.ucar.tn"
__license__ = "MIT"

from FinStoch.processes import (
    StochasticProcess,
    GeometricBrownianMotion,
    MertonJumpDiffusion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    HestonModel,
    ConstantElasticityOfVariance,
    VasicekModel,
    BatesModel,
    VarianceGammaProcess,
)
from FinStoch.bootstrap import BootstrapMonteCarlo

__all__ = [
    "StochasticProcess",
    "GeometricBrownianMotion",
    "MertonJumpDiffusion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "HestonModel",
    "ConstantElasticityOfVariance",
    "VasicekModel",
    "BatesModel",
    "VarianceGammaProcess",
    "BootstrapMonteCarlo",
]

"""FinStoch — A Python library for simulating stochastic processes in finance."""

try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("FinStoch")
except PackageNotFoundError:
    try:
        from FinStoch._version import version as __version__  # type: ignore[no-redef]
    except ImportError:
        __version__ = "0.0.0-unknown"

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
)

__all__ = [
    "StochasticProcess",
    "GeometricBrownianMotion",
    "MertonJumpDiffusion",
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "HestonModel",
    "ConstantElasticityOfVariance",
]

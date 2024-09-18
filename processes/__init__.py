"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

from .gbm import GeometricBrownianMotion
from .merton import MertonModel
from .ou import OrnsteinUhlenbeck
from .cir import CoxIngersollRoss
from .heston import HestonModel

__all__ = ['GeometricBrownianMotion', 'MertonModel', 'OrnsteinUhlenbeck', 'CoxIngersollRoss', 'HestonModel']

__version__ = 'v1.0.0'

__author__ = 'Yosri Ben Halima'

__email__ = 'yosri.benhalima@ept.ucar.tn'

__description__ = 'A financial model simulation library.'

__url__ = 'https://github.com/Yosri-Ben-Halima/FinStoch'

__license__ = 'MIT'

__copyright__ = 'Copyright (c) 2024 Yosri Ben Halima'
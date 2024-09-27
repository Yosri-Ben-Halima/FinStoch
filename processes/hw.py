"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration

class HullWhite:
    """
    HullWhite
    =========

    A class to model the Hull-White model, a stochastic process that allows the mean-reversion level to change over time.
    """
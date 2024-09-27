"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration

class BatesModel:
    """
    BatesModel
    ==========
    
    A class to model the Bates process, a stochastic process that exhibits jump diffusion with stochastic volatility.
    """

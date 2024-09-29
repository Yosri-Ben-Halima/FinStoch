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
    
    Attributes
    ----------
    _S0 : float
        
    """
    
    def __init__(self):
        """
        Initializes a BatesModel object.
        
        Parameters
        ----------
        S0 : float
            The initial value of the asset or variable being modeled.
        mu : float
            The annualized drift or mean return rate of the process.
        sigma : float
            The annualized volatility or standard deviation of the returns.
        lambda_j : float
            The jump intensity parameter.
        mu_j : float
            The mean jump size.
        sigma_j : float
            The volatility of the jump sizes.
        dt : float
            The time step size.
        T : float
            The time duration.
        """

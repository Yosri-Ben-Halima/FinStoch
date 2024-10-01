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
        The initial value of the asset or variable being modeled (protected).
    _mu : float
        The drift rate of the asset (protected).
    _v0 : float
        The initial variance of the asset (protected).
    _kappa : float
        The rate at which the variance reverts to the long-term mean (protected).
    _theta : float
        The long-term mean of the variance process (protected).
    _sigma : float
        The volatility of the volatility process, also known as the vol of vol (protected).
    _rho : float
        The correlation between the Brownian motions driving the asset price and variance processes (protected).
    _lambda_j : float
        The annualized rate of jumps (protected).
    _mu_j : float
        The mean of jump size (protected).
    _sigma_j : float
        The standard deviation of jump size (protected).
    __T : float
        The total time horizon for the simulation, calculated from the date range (private).
    __num_steps : int
        The number of discrete time steps in the simulation, based on the date range (private).
    _num_paths : int
        The number of paths (scenarios) to simulate (protected).
    __dt : float
        The time increment between steps, calculated as T / num_steps (private).
    __t : np.ndarray
        A NumPy array representing the discrete time steps or dates for the simulation (private).

    """
    
    def __init__(self, S0: float, v0: float, mu: float, sigma: float, theta: float, kappa: float, rho: float, lambda_j: float, mu_j: float, sigma_j: float, num_paths: float, start_date: str, end_date: str, granularity: str) -> None:
        """
        Initializes a BatesModel object.

        Parameters
        ----------
        S0 : float
            The initial value of the asset.
        mu : float
            The drift rate of the asset.
        v0 : float
            The initial variance of the asset.
        kappa : float
            The rate of mean reversion for the variance process.
        theta : float
            The long-term mean of the variance process.
        sigma : float
            The volatility of the variance (vol of vol).
        rho : float
            The correlation between the asset price and variance Brownian motions.
        lambda_j : float
            The annualized jump intensity.
        mu_j : float
            The annualized mean of jump size.
        sigma_j : float
            The annualized standard deviation of jump size.
        num_paths : int
            The number of paths to simulate.
        start_date : str
            The start date for the simulation (e.g., '2023-09-01').
        end_date : str
            The end date for the simulation (e.g., '2023-09-01 03:00:00').
        granularity : str
            The time granularity for each step in the simulation (e.g., '10T' for 10 minutes, 'H' for hours).
        """
        self._S0 = S0
        self._v0 = v0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._kappa = kappa
        self._rho = rho
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j

        self._start_date = start_date
        self._end_date = end_date
        self._granularity = granularity
        self.__t = generate_date_range_with_granularity(self._start_date, self._end_date, self._granularity)
        
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
        
        self._num_paths = num_paths

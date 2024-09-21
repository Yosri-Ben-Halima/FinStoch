"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration

class MertonModel:

    """
    MertonModel
    ===========

    A class to simulate a jump diffusion process based on the Merton model.

    Attributes
    ----------
    S0 : float
        Initial value of the variable.
    mu : float
        The rift coefficient.
    sigma : float
        The volatility.
    T : float
        The time horizon (in years) for the simulation.
    num_steps : int
        Number of time steps for the simulation.
    num_paths : int
        Number of paths to simulate.
    dt : float
        The time increment between steps (T / num_steps).
    lambda_j : float
        Jump intensity.
    mu_j : float
        Mean of jump size.
    sigma_j : float
        Standard deviation of jump size.
    start_date : str
        The start date for the simulation. If not provided, time is treated numerically.
    t : np.ndarray
        The time or date range for the simulation steps.

    """

    def __init__(self, S0: float, mu: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float, num_paths: int, start_date: str, end_date: str, granularity: str) -> None:
        """
        Initialize the parameters for the Merton model and set up the time or date steps.

        Parameters
        ----------
        S0 : float
            Initial value of the variable.
        mu : float
            Drift coefficient.
        sigma : float
            Volatility.
        lambda_j : float
            Jump intensity.
        mu_j : float
            Mean of jump size.
        sigma_j : float
            Standard deviation of jump size.
        num_paths : int
            The number of paths to simulate.
        start_date : str
            The start date for the simulation (e.g., '2023-09-01'). 
        end_date : str
            The end date for the simulation (e.g., '2023-09-01'). 
        granularity : str
            The time granularity for each step in the simulation (e.g., 'D' for daily).
        """
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
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

    def simulate(self) -> np.ndarray:
        """
        Simulates a path of the Merton Model jump diffusion model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros((self._num_paths, self.__num_steps))
        S[:, 0] = self._S0

        for t in range(1, self.__num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            N = generate_random_numbers('poisson', self._num_paths, lam=self._lambda_j * self.__dt)
            J = np.zeros(self._num_paths)

            J[N > 0] = generate_random_numbers('normal', np.sum(N > 0), mean=self._mu_j, stddev=self._sigma_j)
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self.__dt + self._sigma * np.sqrt(self.__dt) * Z + J)

        return S

    def plot(self, paths=None, title="Merton Model", ylabel='Value', fig_size: tuple=None, **kwargs):
        """
        Plots the simulated paths of the Merton model.

        Parameters
        ----------
        paths : np.ndarray, optional
            If specified, the function will plot the given paths. Otherwise, it simulates new paths.
        title : str, optional
            Title for the plot. Default is 'Merton Model'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Value'.
        fig_size : tuple, optional
            Size of the figure in inches. Default is None.
        **kwargs
            Additional keyword arguments to pass to the `plot_simulated_paths` function.

        Returns
        -------
        None
        """
        plot_simulated_paths(self.__t, self.simulate, paths, title=title, ylabel=ylabel, fig_size=fig_size, grid=kwargs.get('grid', True))
    
    @property
    def S0(self) -> float:
        return self._S0

    @S0.setter
    def S0(self, value: float) -> None:
        self._S0 = value

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = value

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = value

    @property
    def lambda_j(self) -> float:
        return self._lambda_j

    @lambda_j.setter
    def lambda_j(self, value: float) -> None:
        self._lambda_j = value
    
    @property
    def mu_j(self) -> float:
        return self._mu_j
    
    @mu_j.setter
    def mu_j(self, value: float) -> None:
        self._mu_j = value
    
    @property
    def sigma_j(self) -> float:
        return self._sigma_j
    
    @sigma_j.setter
    def sigma_j(self, value: float) -> None:
        self._sigma_j = value

    @property
    def T(self) -> float:
        return self.__T

    @property
    def num_steps(self) -> int:
        return self.__num_steps
    
    @property
    def num_paths(self) -> int:
        return self._num_paths

    @num_paths.setter
    def num_paths(self, value: int) -> None:
        self._num_paths = value
    
    @property
    def dt(self) -> float:
        return self.__dt
    
    @property
    def t(self) -> np.ndarray:
        return self.__t

    @property
    def start_date(self) -> np.ndarray:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self.__t = generate_date_range_with_granularity(value, self._end_date, self._granularity)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps   
    
    @property
    def end_date(self) -> np.ndarray:
        return self._end_date
    
    @end_date.setter
    def end_date(self, value: str) -> None:
        self._end_date = value
        self.__t = generate_date_range_with_granularity(self._start_date, value, self._granularity)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
    
    @property
    def granularity(self) -> np.ndarray:
        return self._granularity
    
    @granularity.setter
    def granularity(self, value: str) -> None:
        self._granularity = value
        self.__t = generate_date_range_with_granularity(self._start_date, self._end_date, value)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
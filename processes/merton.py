"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range

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

    def __init__(self, S0: float, mu: float, sigma: float, T: float, num_steps: int, num_paths: int, lambda_j: float, mu_j: float, sigma_j: float, start_date: str=None) -> None:
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
        T : float
            Time horizon.
        num_steps : int
            Number of time steps.
        num_paths : int
            Number of simulated paths.
        lambda_j : float
            Jump intensity.
        mu_j : float
            Mean of jump size.
        sigma_j : float
            Standard deviation of jump size.
        start_date : str, optional
            Start date for the simulation. If not provided, time points will be generated using np.linspace.
        """
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T / num_steps
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j
        self._start_date = start_date
        if start_date is not None:
            self._t = generate_date_range(self._start_date, self._T, self._num_steps)
        else :
            self._t = np.linspace(0, self._T, self._num_steps)

    def simulate(self) -> np.ndarray:
        """
        Simulates a path of the Merton Model jump diffusion model.

        Parameters
        ----------
        self : MertonModel
            Instance of MertonModel class.
        
        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            N = generate_random_numbers('poisson', self._num_paths, lam=self._lambda_j * self._dt)
            J = np.zeros(self._num_paths)

            J[N > 0] = generate_random_numbers('normal', np.sum(N > 0), mean=self._mu_j, stddev=self._sigma_j)
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self._dt + self._sigma * np.sqrt(self._dt) * Z + J)

        return S

    def plot(self, paths=None, title="Merton Model", ylabel='Value', **kwargs):
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
        **kwargs
            Additional keyword arguments to pass to the `plot_simulated_paths` function.

        Returns
        -------
        None
        """
        plot_simulated_paths(self._t, self.simulate, paths, title=title, ylabel=ylabel, grid=kwargs.get('grid', True))
    
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
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        self._T = value
        self._dt = value / self._num_steps
        if self._start_date is not None:
            self._t = generate_date_range(self._start_date, value, self._num_steps)
        else :
            self._t = np.linspace(0, value, self._num_steps)

    @property
    def num_steps(self) -> int:
        return self._num_steps
    
    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._num_steps = value
        self._dt = self._T / value
        if self._start_date is not None:
            self._t = generate_date_range(self._start_date, self._T, value)
        else :
            self._t = np.linspace(0, self._T, value)
    
    @property
    def num_paths(self) -> int:
        return self._num_paths

    @num_paths.setter
    def num_paths(self, value: int) -> None:
        self._num_paths = value

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
    def dt(self) -> float:
        return self._dt
    
    @property
    def t(self) -> np.ndarray:
        return self._t
    
    @property
    def start_date(self) -> np.ndarray:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self._t = generate_date_range(value, self._T, self._num_steps)
"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range

class OrnsteinUhlenbeck :

    def __init__(self, S0: float, mu: float, sigma: float, theta: float, T: float, num_steps: float, num_paths: float, start_date: str=None, end_date: str=None) -> None:
        
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
        self._start_date = start_date
        self._end_date = end_date
        if start_date is not None:
            self._t = generate_date_range(self._start_date, self._T, self._num_steps)
        else :
            self._t = np.linspace(0, self._T, self._num_steps)

    def simulate(self) -> float:
        """
        Simulates a path of the Ornstein Uhlenbeck model.

        Parameters
        ----------
        self : OrnsteinUhlenbeck
            Instance of OrnsteinUhlenbeck class.
        
        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros(( self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            drift = self._theta * (self._mu - S[:, t - 1]) * self._dt
            diffusion = self._sigma * np.sqrt(self._dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion

        return S

    def plot(self, paths=None, title="Ornstein Uhlenbeck", ylabel='Value', **kwargs):
        """
        Plots the simulated paths of the Ornstein Uhlenbeck model.

        Parameters
        ----------
        paths : np.ndarray, optional
            If specified, the function will plot the given paths. Otherwise, it simulates new paths.
        title : str, optional
            Title for the plot. Default is 'Ornstein Uhlenbeck'.
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
    def theta(self) -> float:
        return self._theta

    @sigma.setter
    def theta(self, value: float) -> None:
        self._theta = value

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
    def dt(self) -> float:
        return self._dt
    
    @property
    def t(self) -> np.ndarray:
        return self._t
    
    @property
    def start_date(self) -> str:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self._t = generate_date_range(value, self._T, self._num_steps)
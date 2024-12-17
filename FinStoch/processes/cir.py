"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np
from pandas import DatetimeIndex
from FinStoch.utils.random import generate_random_numbers
from FinStoch.utils.plotting import plot_simulated_paths
from FinStoch.utils.timesteps import (
    generate_date_range_with_granularity,
    date_range_duration,
)
from typing import Optional


class CoxIngersollRoss:
    """
    CoxIngersollRoss
    ================

    Class for simulating the Cox-Ingersoll-Ross (CIR) model, often used for modeling non-negative mean-reverting processes.

    Attributes
    ----------
    S0 : float
        The initial value of the stochastic process.
    mu : float
        The long-term mean of the process (also called the reversion level).
    sigma : float
        The volatility parameter of the process.
    theta : float
        The speed of mean reversion.
    start_date : str
        The starting date for the simulation.
    end_date : str
        The ending date for the simulation.
    granularity : str
        The time step granularity for the date range (e.g., 'daily', 'monthly').
    __T : float
        The total duration of the simulation in years.
    __num_steps : int
        The number of time steps in the simulation.
    __dt : float
        The time increment between each step.
    _num_paths : int
        The number of simulation paths to generate.
    __t : np.ndarray
        The array of time steps for the simulation.

    Methods
    -------
    simulate() -> np.ndarray
        Simulates multiple paths of the GBM process and returns the simulated paths as a 2D array.

    plot(paths=None, title="Cox-Ingersoll-Ross", ylabel='Value', **kwargs) -> None
        Plots the simulated paths of the GBM process. If paths are provided, it will plot those paths; otherwise, it will simulate new paths and plot them.

    Properties
    ----------
    S0 :
        Getter and setter for the initial value of the variable.
    mu :
        Getter and setter for the long term mean.
    sigma :
        Getter and setter for the volatility.
    theta :
        Getter and setter for the speed of mean reversion.
    T :
        Getter for the time horizon of the simulation (private attribute).
    num_steps :
        Getter for the number of steps in the simulation (private attribute).
    num_paths :
        Getter and setter for the number of paths to simulate.
    dt :
        Getter for the time increment between steps (private attribute).
    t :
        Getter for the time steps or dates used in the simulation (private attribute).
    start_date :
        Getter and setter for the start date.
    end_date :
        Getter and setter for the end date.
    granularity :
        Getter and setter for the time granularity.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        theta: float,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
    ) -> None:
        """
        Initialize the CIR model with given parameters.

        Parameters
        ----------
        S0 : float
            Initial value of the process.
        mu : float
            Long-term mean to which the process reverts.
        sigma : float
            Volatility parameter of the process.
        theta : float
            Speed of reversion to the mean.
        T : float
            Time horizon for the simulation.
        num_steps : int
            Number of time steps in the simulation.
        num_paths : int
            Number of simulation paths to generate.
        start_date : str
            Starting date for the simulation.
        end_date : str
            Ending date for the simulation.
        granularity : str
            Granularity of time steps (e.g., '10T' for 10 minutes, 'H' for hours).
        """
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta

        self._start_date = start_date
        self._end_date = end_date
        self._granularity = granularity
        self.__t = generate_date_range_with_granularity(
            self._start_date, self._end_date, self._granularity
        )

        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T / self.__num_steps

        self._num_paths = num_paths

    def simulate(self) -> np.ndarray:
        """
        Simulates paths of the CIR model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros((self._num_paths, self.__num_steps))
        S[:, 0] = self._S0

        for t in range(1, self.__num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            drift = self._theta * (self._mu - S[:, t - 1]) * self.__dt
            diffusion = self._sigma * np.sqrt(S[:, t - 1]) * np.sqrt(self.__dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion

            # Ensure non-negativity
            S[:, t] = np.maximum(S[:, t], 0)

        return S

    def plot(
        self,
        paths=None,
        title="Cox-Ingersoll-Ross",
        ylabel="Value",
        fig_size: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Plot the simulated paths of the CIR model.

        Parameters
        ----------
        paths : np.ndarray, optional
            If provided, the method will plot the given paths. Otherwise, it simulates new paths.
        title : str, optional
            Title for the plot. Default is 'CIR Model'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Value'.
        fig_size : tuple, optional
            Size of the figure in inches. Default is None.
        **kwargs
            Additional keyword arguments for the plot, such as grid=True to show the grid.

        Returns
        -------
        None
        """
        plot_simulated_paths(
            self.__t,
            self.simulate,
            paths,
            title=title,
            ylabel=ylabel,
            fig_size=fig_size,
            grid=kwargs.get("grid", True),
        )

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

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = value

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
    def t(self) -> DatetimeIndex:
        return self.__t

    @property
    def start_date(self) -> str:
        return self._start_date

    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self.__t = generate_date_range_with_granularity(
            value, self._end_date, self._granularity
        )
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T / self.__num_steps

    @property
    def end_date(self) -> str:
        return self._end_date

    @end_date.setter
    def end_date(self, value: str) -> None:
        self._end_date = value
        self.__t = generate_date_range_with_granularity(
            self._start_date, value, self._granularity
        )
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T / self.__num_steps

    @property
    def granularity(self) -> str:
        return self._granularity

    @granularity.setter
    def granularity(self, value: str) -> None:
        self._granularity = value
        self.__t = generate_date_range_with_granularity(
            self._start_date, self._end_date, value
        )
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T / self.__num_steps

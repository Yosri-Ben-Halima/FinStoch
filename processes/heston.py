"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np
from pandas import DatetimeIndex
from typing import Optional, Tuple
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration


class HestonModel:
    """
    HestonModel
    ===========

    A class to model the Heston stochastic volatility process.

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

    Methods
    -------
    simulate() -> np.ndarray
        Simulates multiple paths of the Heston process for both asset prices and volatility.
    plot(paths=None, title="Heston Model Simulation", ylabel='Asset Price', **kwargs) -> None
        Plots the simulated paths of the Heston process. If paths are provided, it will plot those paths; otherwise, it will simulate new paths and plot them.

    Properties
    ----------
    S0 :
        Getter and setter for the initial value of the asset.
    mu :
        Getter and setter for the drift rate.
    v0 :
        Getter and setter for the initial variance.
    kappa :
        Getter and setter for the mean reversion rate of the variance.
    theta :
        Getter and setter for the long-term mean of the variance.
    sigma :
        Getter and setter for the volatility of the volatility.
    rho :
        Getter and setter for the correlation between Brownian motions.
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
    """

    def __init__(
        self,
        S0: float,
        v0: float,
        mu: float,
        sigma: float,
        theta: float,
        kappa: float,
        rho: float,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
    ) -> None:
        """
        Initialize the Heston stochastic volatility process.

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

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Heston model.

        Returns
        -------
        np.ndarray
            A tuple of two 2D arrays representing simulated asset prices and volatilities, where each row represents a simulated path.
            The first array has the shape (num_paths, num_steps) for process values (e.g. asset prices), and the second array has the same shape for volatilities.
        """
        S = np.zeros((self._num_paths, self.__num_steps))
        S[:, 0] = self._S0

        v = np.zeros((self._num_paths, self.__num_steps))
        v[:, 0] = self._v0

        for t in range(1, self.__num_steps):
            Xs, Xv = (
                generate_random_numbers("normal", self._num_paths, mean=0, stddev=1),
                generate_random_numbers("normal", self._num_paths, mean=0, stddev=1),
            )
            L = np.array([[1, 0], [self._rho, np.sqrt(1 - self._rho**2)]])

            X = np.dot(L, np.array([Xs, Xv]))
            Ws = X[0]
            Wv = X[1]

            v[:, t] = np.maximum(
                v[:, t - 1]
                + self._kappa * (self._theta - v[:, t - 1]) * self.__dt
                + self._sigma * np.sqrt(v[:, t - 1]) * np.sqrt(self.__dt) * Wv,
                0,
            )
            S[:, t] = S[:, t - 1] * np.exp(
                (self._mu - 0.5 * v[:, t - 1]) * self.__dt
                + np.sqrt(v[:, t - 1]) * np.sqrt(self.__dt) * Ws
            )

        return S, v

    def plot(
        self,
        paths=None,
        title: str = "Heston Model",
        ylabel="Value",
        fig_size: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Plot the simulated paths of the Heston Model.

        Parameters
        ----------
        paths : np.ndarray, optional
            If provided, the method will plot the given paths. Otherwise, it simulates new paths.
        title : str, optional
            Title for the plot. Default is 'Heston Model'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Value'.
        fig_size : tuple, optional
            Size of the figure in inches. Default is None.
        **kwargs :
            Additional keyword arguments for customizing the plot:

            'variance': bool, optional

                If True, plots the variance paths of the Heston model instead of the asset price paths.
                Default is False.

            'grid': bool, optional

                If True, displays gridlines on the plot. Default is True.

        Returns
        -------
        None
        """

        if kwargs.get("variance", False) is True:
            plot_simulated_paths(
                self.__t,
                self.simulate,
                paths,
                title=title,
                ylabel=ylabel,
                fig_size=fig_size,
                **kwargs,
            )
        else:
            plot_simulated_paths(
                self.__t,
                self.simulate,
                paths,
                title=title,
                ylabel=ylabel,
                fig_size=fig_size,
                **kwargs,
            )

    @property
    def S0(self) -> float:
        return self._S0

    @S0.setter
    def S0(self, value: float) -> None:
        self._S0 = value

    @property
    def v0(self) -> float:
        return self._v0

    @v0.setter
    def v0(self, value: float) -> None:
        self._v0 = value

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
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, value: float) -> None:
        self._kappa = value

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        self._rho = value

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

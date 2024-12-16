"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np
from pandas import DatetimeIndex
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration
from typing import Optional


class MertonJumpDiffusion:
    """
    MertonJumpDiffusion
    ===========

    A class to simulate a jump diffusion process based on the Merton model.

    This model extends the Black-Scholes model by incorporating jumps, which are modeled using a Poisson process.
    The underlying asset price evolves according to a combination of continuous Brownian motion and discrete jumps.

    Attributes
    ----------
    _S0 : float
        Initial value of the asset or process at the start date.
    _mu : float
        The annualized drift coefficient (protected).
    _sigma : float
        The annualized volatility (protected).
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
    _start_date : str
        The start date for the simulation, provided as a string (protected).
    _end_date : str
        The end date for the simulation, provided as a string (protected).
    _granularity : str
        The time granularity for simulation steps, given as a Pandas frequency string (protected).
    __t : np.ndarray
        A NumPy array representing the discrete time steps or dates for the simulation (private).

    Methods
    -------
    simulate() -> np.ndarray
        Simulates multiple paths of the Merton model and returns the simulated paths as a 2D array.

    plot(paths=None, title="Merton Model", ylabel='Value', **kwargs) -> None
        Plots the simulated paths of the Merton model. If paths are provided, it will plot those paths; otherwise, it will simulate new paths and plot them.

    Properties
    ----------
    S0 :
        Getter and setter for the initial value of the process.
    mu :
        Getter and setter for the drift (expected return).
    sigma :
        Getter and setter for the volatility of the process.
    lambda_j :
        Getter and setter for the jump intensity parameter.
    mu_j :
        Getter and setter for the mean jump size.
    sigma_j :
        Getter and setter for the standard deviation of jump sizes.
    num_paths :
        Getter and setter for the number of paths to simulate.
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
        lambda_j: float,
        mu_j: float,
        sigma_j: float,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
    ) -> None:
        """
        Initialize the parameters for the Merton model and set up the time or date steps.

        Parameters
        ----------
        S0 : float
            Initial value of the variable.
        mu : float
            The annualized drift coefficient.
        sigma : float
            The annualized volatility.
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
        self.__t = generate_date_range_with_granularity(
            self._start_date, self._end_date, self._granularity
        )

        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T / self.__num_steps

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

        k = np.exp(self._mu_j + 0.5 * self._sigma_j**2) - 1

        for t in range(1, self.__num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            N = generate_random_numbers(
                "poisson", self._num_paths, lam=self._lambda_j * self.__dt
            )
            J = np.zeros(self._num_paths)

            J[N > 0] = generate_random_numbers(
                "normal", int(np.sum(N > 0)), mean=self._mu_j, stddev=self._sigma_j
            )
            S[:, t] = S[:, t - 1] * np.exp(
                (self._mu - 0.5 * self._sigma**2 - self._lambda_j * k) * self.__dt
                + self._sigma * np.sqrt(self.__dt) * Z
                + J
            )

        return S

    def plot(
        self,
        paths=None,
        title="Merton Model",
        ylabel="Value",
        fig_size: Optional[tuple] = None,
        **kwargs,
    ):
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

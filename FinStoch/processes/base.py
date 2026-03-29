"""Base class for all stochastic process simulators."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from pandas import DatetimeIndex

from FinStoch.utils.plotting import plot_simulated_paths
from FinStoch.utils.timesteps import (
    generate_date_range_with_granularity,
    date_range_duration,
)


class StochasticProcess(ABC):
    """Abstract base class for stochastic process simulators.

    Provides shared initialization, time grid management, and plotting
    for all Euler-Maruyama discretized stochastic processes.

    Parameters
    ----------
    S0 : float
        The initial value of the process.
    mu : float
        The drift coefficient.
    sigma : float
        The volatility coefficient.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation (e.g., '2023-09-01').
    end_date : str
        The end date for the simulation (e.g., '2023-12-31').
    granularity : str
        The time granularity for each step (e.g., 'D', 'H', '10T').
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
    ) -> None:
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._num_paths = num_paths
        self._start_date = start_date
        self._end_date = end_date
        self._granularity = granularity
        self._recalculate_time_grid()

    def _recalculate_time_grid(self) -> None:
        """Recompute time grid attributes from date range and granularity."""
        self._t = generate_date_range_with_granularity(self._start_date, self._end_date, self._granularity)
        self._T = date_range_duration(self._t)
        self._num_steps = len(self._t)
        self._dt = self._T / self._num_steps

    @abstractmethod
    def simulate(self) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Simulate paths of the stochastic process.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            A 2D array of shape (num_paths, num_steps), or a tuple of two
            such arrays for models with multiple outputs (e.g., Heston).
        """
        ...

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Simulated Paths",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated paths.

        Parameters
        ----------
        paths : np.ndarray, optional
            Pre-computed paths to plot. If None, calls simulate().
        title : str
            Plot title.
        ylabel : str
            Y-axis label.
        fig_size : tuple, optional
            Figure size in inches.
        **kwargs
            Additional keyword arguments passed to plot_simulated_paths.
        """
        plot_simulated_paths(
            self._t,
            self.simulate,
            paths,
            title=title,
            ylabel=ylabel,
            fig_size=fig_size,
            grid=kwargs.get("grid", True),
        )

    # --- Shared properties ---

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

    @property
    def num_steps(self) -> int:
        return self._num_steps

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
    def t(self) -> DatetimeIndex:
        return self._t

    @property
    def start_date(self) -> str:
        return self._start_date

    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self._recalculate_time_grid()

    @property
    def end_date(self) -> str:
        return self._end_date

    @end_date.setter
    def end_date(self, value: str) -> None:
        self._end_date = value
        self._recalculate_time_grid()

    @property
    def granularity(self) -> str:
        return self._granularity

    @granularity.setter
    def granularity(self, value: str) -> None:
        self._granularity = value
        self._recalculate_time_grid()

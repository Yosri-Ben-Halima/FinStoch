"""Ornstein-Uhlenbeck process."""

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.random import generate_random_numbers


class OrnsteinUhlenbeck(StochasticProcess):
    """Ornstein-Uhlenbeck mean-reverting process simulator.

    Models a process following the SDE:
        dS = theta * (mu - S) * dt + sigma * dW

    Parameters
    ----------
    S0 : float
        The initial value of the process.
    mu : float
        The long-term mean level to which the process reverts.
    sigma : float
        The volatility of the process.
    theta : float
        The speed of mean reversion.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation.
    end_date : str
        The end date for the simulation.
    granularity : str
        The time granularity for each step.
    business_days : bool, optional
        If True, use business days instead of calendar days. Default is False.
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
        business_days: bool = False,
    ) -> None:
        self._theta = theta
        super().__init__(S0, mu, sigma, num_paths, start_date, end_date, granularity, business_days)

    def simulate(self, seed: int | None = None) -> np.ndarray:
        """Simulate paths of the Ornstein-Uhlenbeck model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        if seed is not None:
            np.random.seed(seed)
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            drift = self._theta * (self._mu - S[:, t - 1]) * self._dt
            diffusion = self._sigma * np.sqrt(self._dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Ornstein Uhlenbeck",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Ornstein-Uhlenbeck paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = value

"""Geometric Brownian Motion process."""

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.random import generate_random_numbers


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion (GBM) process simulator.

    Models an asset price following the SDE:
        dS = mu * S * dt + sigma * S * dW

    Parameters
    ----------
    S0 : float
        The initial value of the asset.
    mu : float
        The annualized drift coefficient.
    sigma : float
        The annualized volatility coefficient.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation (e.g., '2023-09-01').
    end_date : str
        The end date for the simulation (e.g., '2023-12-31').
    granularity : str
        The time granularity for each step (e.g., '10T' for 10 minutes, 'H' for hours).
    """

    def simulate(self) -> np.ndarray:
        """Simulate paths of the GBM model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            S[:, t] = S[:, t - 1] * np.exp((self._mu - 0.5 * self._sigma**2) * self._dt + self._sigma * np.sqrt(self._dt) * Z)

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Geometric Brownian Motion",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated GBM paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

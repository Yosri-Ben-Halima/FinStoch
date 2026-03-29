"""Merton Jump Diffusion process."""

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.random import generate_random_numbers


class MertonJumpDiffusion(StochasticProcess):
    """Merton Jump Diffusion process simulator.

    Extends GBM by incorporating Poisson-distributed jumps:
        dS = (mu - lambda_j * k) * S * dt + sigma * S * dW + J * S * dN

    Parameters
    ----------
    S0 : float
        Initial value of the asset.
    mu : float
        The annualized drift coefficient.
    sigma : float
        The annualized volatility.
    lambda_j : float
        The annualized jump intensity.
    mu_j : float
        The mean of jump size.
    sigma_j : float
        The standard deviation of jump size.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation.
    end_date : str
        The end date for the simulation.
    granularity : str
        The time granularity for each step.
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
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j
        super().__init__(S0, mu, sigma, num_paths, start_date, end_date, granularity)

    def simulate(self) -> np.ndarray:
        """Simulate paths of the Merton Jump Diffusion model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        k = np.exp(self._mu_j + 0.5 * self._sigma_j**2) - 1

        for t in range(1, self._num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            N = generate_random_numbers("poisson", self._num_paths, lam=self._lambda_j * self._dt)
            J = np.zeros(self._num_paths)

            J[N > 0] = generate_random_numbers("normal", int(np.sum(N > 0)), mean=self._mu_j, stddev=self._sigma_j)
            S[:, t] = S[:, t - 1] * np.exp(
                (self._mu - 0.5 * self._sigma**2 - self._lambda_j * k) * self._dt + self._sigma * np.sqrt(self._dt) * Z + J
            )

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Merton Model",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Merton Jump Diffusion paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

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

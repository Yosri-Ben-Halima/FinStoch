"""Constant Elasticity of Variance process."""

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.random import generate_random_numbers


class ConstantElasticityOfVariance(StochasticProcess):
    """Constant Elasticity of Variance (CEV) process simulator.

    Models an asset price following the SDE:
        dS = mu * S * dt + sigma * S^gamma * dW

    Parameters
    ----------
    S0 : float
        The initial value of the asset.
    mu : float
        The annualized drift coefficient.
    sigma : float
        The annualized volatility coefficient.
    gamma : float
        The elasticity parameter.
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
        gamma: float,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
        business_days: bool = False,
    ) -> None:
        self._gamma = gamma
        super().__init__(S0, mu, sigma, num_paths, start_date, end_date, granularity, business_days)

    def simulate(self) -> np.ndarray:
        """Simulate paths of the CEV model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers("normal", self._num_paths, mean=0, stddev=1)
            S[:, t] = (
                S[:, t - 1]
                + self._mu * S[:, t - 1] * self._dt
                + self._sigma * (S[:, t - 1] ** self._gamma) * np.sqrt(self._dt) * Z
            )

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Constant Elasticity of Variance",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated CEV paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = value

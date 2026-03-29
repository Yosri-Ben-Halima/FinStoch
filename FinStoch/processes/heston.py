"""Heston stochastic volatility model."""

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.plotting import plot_simulated_paths
from FinStoch.utils.random import generate_random_numbers


class HestonModel(StochasticProcess):
    """Heston stochastic volatility model simulator.

    Models an asset price with stochastic variance:
        dS = mu * S * dt + sqrt(v) * S * dW_s
        dv = kappa * (theta - v) * dt + sigma * sqrt(v) * dW_v
        corr(dW_s, dW_v) = rho

    Parameters
    ----------
    S0 : float
        The initial value of the asset.
    v0 : float
        The initial variance.
    mu : float
        The drift rate of the asset.
    sigma : float
        The volatility of the variance (vol of vol).
    theta : float
        The long-term mean of the variance process.
    kappa : float
        The rate of mean reversion for the variance.
    rho : float
        The correlation between asset and variance Brownian motions.
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
        business_days: bool = False,
    ) -> None:
        self._v0 = v0
        self._theta = theta
        self._kappa = kappa
        self._rho = rho
        super().__init__(S0, mu, sigma, num_paths, start_date, end_date, granularity, business_days)

    def simulate(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate paths of the Heston model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (S, v) of 2D arrays of shape (num_paths, num_steps)
            for asset prices and variance paths respectively.
        """
        if seed is not None:
            np.random.seed(seed)
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        v = np.zeros((self._num_paths, self._num_steps))
        v[:, 0] = self._v0

        for t in range(1, self._num_steps):
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
                + self._kappa * (self._theta - v[:, t - 1]) * self._dt
                + self._sigma * np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Wv,
                0,
            )
            S[:, t] = S[:, t - 1] * np.exp(
                (self._mu - 0.5 * v[:, t - 1]) * self._dt + np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Ws
            )

        return S, v

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Heston Model",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Heston paths.

        Parameters
        ----------
        paths : np.ndarray, optional
            Pre-computed paths to plot.
        title : str
            Plot title.
        ylabel : str
            Y-axis label.
        fig_size : tuple, optional
            Figure size in inches.
        **kwargs
            Additional keyword arguments. Pass ``variance=True`` to plot
            variance paths instead of asset prices.
        """
        plot_simulated_paths(
            self._t,
            self.simulate,
            paths,
            title=title,
            ylabel=ylabel,
            fig_size=fig_size,
            **kwargs,
        )

    @property
    def v0(self) -> float:
        return self._v0

    @v0.setter
    def v0(self, value: float) -> None:
        self._v0 = value

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

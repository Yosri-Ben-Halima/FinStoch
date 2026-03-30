"""Bates stochastic volatility jump-diffusion model."""

import warnings
from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.plotting import plot_simulated_paths


@dataclass(kw_only=True)
class BatesModel(StochasticProcess):
    """Bates stochastic volatility jump-diffusion model simulator.

    Combines the Heston stochastic volatility model with Merton-style
    jumps in the asset price:
        dS = (mu - lambda_j * k) * S * dt + sqrt(v) * S * dW_s + J * S * dN
        dv = kappa * (theta - v) * dt + sigma * sqrt(v) * dW_v
        corr(dW_s, dW_v) = rho
    """

    v0: float
    theta: float
    kappa: float
    rho: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> tuple[np.ndarray, np.ndarray]:
        """Simulate paths of the Bates model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' for Euler-Maruyama, 'milstein' for Milstein scheme.
            Milstein adds 0.25 * sigma^2 * (Wv^2 - 1) * dt to the
            variance process. Jumps are unaffected by the scheme.
            'exact' falls back to 'euler' with a warning (no
            closed-form path simulation).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (S, v) of 2D arrays of shape (num_paths, num_steps)
            for asset prices and variance paths respectively.
        """
        self._validate_method(method)
        if method == "exact":
            warnings.warn(
                "Exact transition density is not available for the Bates model. Falling back to Euler-Maruyama.",
                stacklevel=2,
            )
            method = "euler"
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0

        v = np.zeros((self.num_paths, self._num_steps))
        v[:, 0] = self.v0

        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

        L = np.array([[1, 0], [self.rho, np.sqrt(1 - self.rho**2)]])
        Xs_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))
        Xv_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))
        N_all = np.random.poisson(self.lambda_j * self._dt, (self.num_paths, self._num_steps - 1))
        J_all = np.random.normal(self.mu_j, self.sigma_j, (self.num_paths, self._num_steps - 1))

        for t in range(1, self._num_steps):
            X = np.dot(L, np.array([Xs_all[:, t - 1], Xv_all[:, t - 1]]))
            Ws = X[0]
            Wv = X[1]

            v[:, t] = (
                v[:, t - 1]
                + self.kappa * (self.theta - v[:, t - 1]) * self._dt
                + self.sigma * np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Wv
            )

            if method == "milstein":
                v[:, t] += 0.25 * self.sigma**2 * (Wv**2 - 1) * self._dt

            v[:, t] = np.maximum(v[:, t], 0)

            N = N_all[:, t - 1]
            J = np.where(N > 0, J_all[:, t - 1], 0.0)

            S[:, t] = S[:, t - 1] * np.exp(
                (self.mu - 0.5 * v[:, t - 1] - self.lambda_j * k) * self._dt
                + np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Ws
                + J
            )

        return S, v

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Bates Model",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Bates paths.

        Pass ``variance=True`` to plot variance paths instead of prices.
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

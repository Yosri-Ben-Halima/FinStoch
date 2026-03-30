"""Merton Jump Diffusion process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class MertonJumpDiffusion(StochasticProcess):
    """Merton Jump Diffusion process simulator.

    Extends GBM by incorporating Poisson-distributed jumps:
        dS = (mu - lambda_j * k) * S * dt + sigma * S * dW + J * S * dN
    """

    lambda_j: float
    mu_j: float
    sigma_j: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the Merton Jump Diffusion model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' uses the exact log-normal diffusion (default).
            'milstein' uses Euler-Milstein for the diffusion component;
            jumps remain multiplicative and are unaffected.
            'exact' is accepted as an alias for 'euler' (both use the
            exact log-normal diffusion with exact Poisson jumps).

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        self._validate_method(method)
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0

        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

        Z_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))
        N_all = np.random.poisson(self.lambda_j * self._dt, (self.num_paths, self._num_steps - 1))
        J_all = np.random.normal(self.mu_j, self.sigma_j, (self.num_paths, self._num_steps - 1))

        for t in range(1, self._num_steps):
            Z = Z_all[:, t - 1]
            N = N_all[:, t - 1]
            J = np.where(N > 0, J_all[:, t - 1], 0.0)

            if method == "milstein":
                S[:, t] = (
                    S[:, t - 1]
                    + (self.mu - self.lambda_j * k) * S[:, t - 1] * self._dt
                    + self.sigma * S[:, t - 1] * np.sqrt(self._dt) * Z
                    + 0.5 * self.sigma**2 * S[:, t - 1] * (Z**2 - 1) * self._dt
                    + S[:, t - 1] * (np.exp(J) - 1)
                )
            else:
                S[:, t] = S[:, t - 1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2 - self.lambda_j * k) * self._dt + self.sigma * np.sqrt(self._dt) * Z + J
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

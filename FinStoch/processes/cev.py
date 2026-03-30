"""Constant Elasticity of Variance process."""

import warnings
from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class ConstantElasticityOfVariance(StochasticProcess):
    """Constant Elasticity of Variance (CEV) process simulator.

    Models an asset price following the SDE:
        dS = mu * S * dt + sigma * S^gamma * dW
    """

    gamma: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the CEV model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' for Euler-Maruyama, 'milstein' for Milstein scheme.
            Milstein adds 0.5 * sigma^2 * gamma * S^(2*gamma-1) * (Z^2-1) * dt.
            'exact' falls back to 'euler' with a warning (no closed form).

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).
        """
        self._validate_method(method)
        if method == "exact":
            warnings.warn(
                "Exact transition density is not available for the CEV process. Falling back to Euler-Maruyama.",
                stacklevel=2,
            )
            method = "euler"
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0
        Z_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))

        for t in range(1, self._num_steps):
            Z = Z_all[:, t - 1]
            S[:, t] = (
                S[:, t - 1]
                + self.mu * S[:, t - 1] * self._dt
                + self.sigma * (S[:, t - 1] ** self.gamma) * np.sqrt(self._dt) * Z
            )

            if method == "milstein":
                S[:, t] += 0.5 * self.sigma**2 * self.gamma * (S[:, t - 1] ** (2 * self.gamma - 1)) * (Z**2 - 1) * self._dt

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

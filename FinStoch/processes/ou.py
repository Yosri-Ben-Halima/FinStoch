"""Ornstein-Uhlenbeck process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class OrnsteinUhlenbeck(StochasticProcess):
    """Ornstein-Uhlenbeck mean-reverting process simulator.

    Models a process following the SDE:
        dS = theta * (mu - S) * dt + sigma * dW
    """

    theta: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the Ornstein-Uhlenbeck model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' or 'milstein'. Both produce identical results for OU
            since the diffusion is constant (dg/dS = 0).
            'exact' uses the closed-form Gaussian transition density.

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
        Z_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))

        if method == "exact":
            decay = np.exp(-self.theta * self._dt)
            mean_weight = 1 - decay
            std_exact = self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * self._dt)) / (2 * self.theta))
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                S[:, t] = S[:, t - 1] * decay + self.mu * mean_weight + std_exact * Z
        else:
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                drift = self.theta * (self.mu - S[:, t - 1]) * self._dt
                diffusion = self.sigma * np.sqrt(self._dt) * Z
                S[:, t] = S[:, t - 1] + drift + diffusion
                # Milstein correction is zero (constant diffusion, dg/dS = 0)

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

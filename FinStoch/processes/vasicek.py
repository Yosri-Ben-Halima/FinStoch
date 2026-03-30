"""Vasicek interest rate model."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class VasicekModel(StochasticProcess):
    """Vasicek mean-reverting interest rate model simulator.

    Models a short rate following the SDE:
        dr = a * (b - r) * dt + sigma * dW

    Mathematically equivalent to the Ornstein-Uhlenbeck process but
    uses interest rate conventions: ``a`` for mean reversion speed
    and ``mu`` for the long-term mean level ``b``.
    """

    a: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the Vasicek model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' or 'milstein'. Both produce identical results for
            Vasicek since the diffusion is constant (dg/dr = 0).

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

        for t in range(1, self._num_steps):
            Z = Z_all[:, t - 1]
            drift = self.a * (self.mu - S[:, t - 1]) * self._dt
            diffusion = self.sigma * np.sqrt(self._dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion
            # Milstein correction is zero (constant diffusion, dg/dr = 0)

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Vasicek Model",
        ylabel: str = "Interest Rate",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Vasicek paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

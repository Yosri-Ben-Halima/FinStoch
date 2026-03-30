"""Cox-Ingersoll-Ross process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class CoxIngersollRoss(StochasticProcess):
    """Cox-Ingersoll-Ross (CIR) mean-reverting process simulator.

    Models a non-negative process following the SDE:
        dS = theta * (mu - S) * dt + sigma * sqrt(S) * dW
    """

    theta: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the CIR model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' for Euler-Maruyama, 'milstein' for Milstein scheme.
            Milstein adds a correction of 0.25 * sigma^2 * (Z^2 - 1) * dt.
            'exact' uses the non-central chi-squared transition density.

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

        if method == "exact":
            exp_decay = np.exp(-self.theta * self._dt)
            c = self.sigma**2 * (1 - exp_decay) / (4 * self.theta)
            df = 4 * self.theta * self.mu / self.sigma**2
            for t in range(1, self._num_steps):
                nc = S[:, t - 1] * exp_decay / c
                S[:, t] = c * np.random.noncentral_chisquare(df, nc)
        else:
            Z_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                drift = self.theta * (self.mu - S[:, t - 1]) * self._dt
                diffusion = self.sigma * np.sqrt(S[:, t - 1]) * np.sqrt(self._dt) * Z
                S[:, t] = S[:, t - 1] + drift + diffusion

                if method == "milstein":
                    S[:, t] += 0.25 * self.sigma**2 * (Z**2 - 1) * self._dt

                S[:, t] = np.maximum(S[:, t], 0)

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Cox-Ingersoll-Ross",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated CIR paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

"""Variance Gamma process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class VarianceGammaProcess(StochasticProcess):
    """Variance Gamma process simulator.

    Models an asset price via a time-changed Brownian motion:
        S(t) = S0 * exp((mu + omega) * t + theta * G(t) + sigma * W(G(t)))

    where G(t) is a Gamma process with variance rate ``nu``, and
    ``omega = (1/nu) * ln(1 - theta*nu - sigma^2*nu/2)`` is the
    drift correction ensuring the martingale property.
    """

    theta: float
    nu: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the Variance Gamma process.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            Only 'euler' is supported. The Variance Gamma process uses
            a time-changed Brownian motion, not Euler-Maruyama
            discretization, so the Milstein scheme is not applicable.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps).

        Raises
        ------
        ValueError
            If method is 'milstein'.
        """
        self._validate_method(method)
        if method == "milstein":
            raise ValueError(
                "Milstein scheme is not applicable to the Variance Gamma process (not an Euler-Maruyama discretization)."
            )
        if seed is not None:
            np.random.seed(seed)

        log_arg = 1 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu
        if log_arg <= 0:
            raise ValueError(f"Invalid parameters: 1 - theta*nu - 0.5*sigma^2*nu must be positive, got {log_arg:.6f}.")
        omega = (1 / self.nu) * np.log(log_arg)

        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0

        G_all = np.random.gamma(self._dt / self.nu, self.nu, (self.num_paths, self._num_steps - 1))
        Z_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))

        for t in range(1, self._num_steps):
            g = G_all[:, t - 1]
            z = Z_all[:, t - 1]
            X = self.theta * g + self.sigma * np.sqrt(g) * z
            S[:, t] = S[:, t - 1] * np.exp((self.mu + omega) * self._dt + X)

        return S

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Variance Gamma Process",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Variance Gamma paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

"""Geometric Brownian Motion process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion (GBM) process simulator.

    Models an asset price following the SDE:
        dS = mu * S * dt + sigma * S * dW
    """

    def simulate(self, seed: int | None = None, method: str = "euler") -> np.ndarray:
        """Simulate paths of the GBM model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' uses the exact log-normal solution (default).
            'milstein' uses the Euler-Milstein additive discretization.
            'exact' is accepted as an alias for 'euler' (both use the
            exact log-normal transition).

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
            if method == "milstein":
                S[:, t] = (
                    S[:, t - 1]
                    + self.mu * S[:, t - 1] * self._dt
                    + self.sigma * S[:, t - 1] * np.sqrt(self._dt) * Z
                    + 0.5 * self.sigma**2 * S[:, t - 1] * (Z**2 - 1) * self._dt
                )
            else:
                S[:, t] = S[:, t - 1] * np.exp((self.mu - 0.5 * self.sigma**2) * self._dt + self.sigma * np.sqrt(self._dt) * Z)

        return S

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate GBM parameters from observed prices via exact MLE.

        Log-returns are i.i.d. Normal((mu - sigma^2/2)*dt, sigma^2*dt).

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu' (drift) and 'sigma' (volatility).

        References
        ----------
        Cont, R. & Tankov, P. (2004). Financial Modelling with Jump
        Processes, Ch. 7.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        log_returns = np.diff(np.log(data))
        sigma_hat = float(np.std(log_returns, ddof=1) / np.sqrt(dt))
        mu_hat = float(np.mean(log_returns) / dt + 0.5 * sigma_hat**2)
        return {"mu": mu_hat, "sigma": sigma_hat}

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Geometric Brownian Motion",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated GBM paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

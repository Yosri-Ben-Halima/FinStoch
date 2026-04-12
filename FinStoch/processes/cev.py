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

    def simulate(self, seed: int | None = None, method: str = "euler", antithetic: bool = False) -> np.ndarray:
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
        Z_all = self._generate_normals((self.num_paths, self._num_steps - 1), antithetic)

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

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate CEV parameters via quasi-MLE with profile optimization.

        For each candidate gamma, (mu, sigma) are estimated in closed
        form via weighted OLS. The profile quasi-log-likelihood is then
        maximized over gamma using bounded scalar optimization.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'gamma'.

        References
        ----------
        Chan, K.C., Karolyi, G.A., Longstaff, F.A. & Sanders, A.B.
        (1992). An empirical comparison of alternative models of the
        short-term interest rate. Journal of Finance, 47(3), 1209-1227.
        """
        from scipy.optimize import minimize_scalar

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        increments = np.diff(data)
        S = np.maximum(data[:-1], 1e-10)
        n = len(increments)
        sqrt_dt = np.sqrt(dt)

        def profile_neg_loglik(gamma_val: float) -> float:
            W = S**gamma_val
            u = increments / (W * sqrt_dt)
            v = S * sqrt_dt / W
            mu_val = np.sum(u * v) / np.sum(v**2)
            residuals = u - mu_val * v
            sigma_sq = np.mean(residuals**2)
            if sigma_sq <= 0:
                return 1e30
            return float(0.5 * n * np.log(sigma_sq) + np.sum(np.log(W)))

        result = minimize_scalar(profile_neg_loglik, bounds=(0.01, 2.0), method="bounded")
        gamma_hat = float(result.x)

        # Re-extract mu and sigma at optimal gamma
        W = S**gamma_hat
        u = increments / (W * sqrt_dt)
        v = S * sqrt_dt / W
        mu_hat = float(np.sum(u * v) / np.sum(v**2))
        sigma_hat = float(np.sqrt(np.mean((u - mu_hat * v) ** 2)))

        return {"mu": mu_hat, "sigma": sigma_hat, "gamma": gamma_hat}

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

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

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252, **kwargs: object) -> dict[str, float]:
        """Estimate Bates parameters via two-stage calibration.

        Stage 1: Estimate stochastic volatility parameters using the
        Heston calibration procedure. Stage 2: Extract jump parameters
        from residuals via the EM algorithm.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).
        rv_window : int, optional
            Rolling window for realized variance. Default is 21.
        max_iter : int, optional
            Maximum EM iterations for jump estimation. Default is 200.
        tol : float, optional
            EM convergence tolerance. Default is 1e-8.

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'v0', 'theta',
            'kappa', 'rho', 'lambda_j', 'mu_j', 'sigma_j'.

        References
        ----------
        Bates, D.S. (1996). Jumps and stochastic volatility. Review
        of Financial Studies, 9(1), 69-107.
        """
        import pandas as pd
        from scipy.stats import norm

        from FinStoch.processes.heston import HestonModel

        rv_window = int(kwargs.get("rv_window", 21))  # type: ignore[call-overload]
        max_iter = int(kwargs.get("max_iter", 200))  # type: ignore[call-overload]
        tol = float(kwargs.get("tol", 1e-8))  # type: ignore[call-overload,arg-type]

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < rv_window + 10:
            raise ValueError(f"data must be a 1D array with at least {rv_window + 10} observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        # Stage 1: Heston calibration for SV component
        heston_params = HestonModel.calibrate(data, dt=dt, rv_window=rv_window)

        # Compute residuals after removing diffusive component
        log_returns = np.diff(np.log(data))
        rv = pd.Series(log_returns**2).rolling(rv_window).mean().dropna().values / dt
        rv = np.maximum(rv, 1e-10)

        # Align returns with realized variance
        r_aligned = log_returns[rv_window - 1 :]
        rv_aligned = rv[: len(r_aligned)]
        min_len = min(len(r_aligned), len(rv_aligned))
        r_aligned = r_aligned[:min_len]
        rv_aligned = rv_aligned[:min_len]

        expected_diffusive = (heston_params["mu"] - 0.5 * rv_aligned) * dt
        residuals = r_aligned - expected_diffusive
        n = len(residuals)

        # Stage 2: EM for jump component on residuals
        lambda_j = 0.1
        mu_j = 0.0
        sigma_j = max(float(np.std(residuals)), 1e-6)

        prev_ll = -np.inf
        for _ in range(max_iter):
            var_nj = rv_aligned * dt
            std_nj = np.sqrt(np.maximum(var_nj, 1e-20))
            std_j = np.sqrt(np.maximum(var_nj + sigma_j**2, 1e-20))

            p_nj = np.maximum(1 - lambda_j * dt, 1e-300) * norm.pdf(residuals, 0.0, std_nj)
            p_j = np.maximum(lambda_j * dt, 1e-300) * norm.pdf(residuals, mu_j, std_j)
            total = p_nj + p_j + 1e-300
            tau = p_j / total

            ll = float(np.sum(np.log(total)))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            tau_sum = float(np.sum(tau))
            lambda_j = max(tau_sum / (n * dt), 1e-10)

            if tau_sum > 1e-10:
                w_j = tau / tau_sum
                mu_j = float(np.sum(w_j * residuals))
                sigma_j = max(float(np.sqrt(np.sum(w_j * (residuals - mu_j) ** 2))), 1e-10)
            else:
                mu_j = 0.0
                sigma_j = 1e-6

        # Adjust mu for jump compensator
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        mu_hat = heston_params["mu"] + lambda_j * k

        return {
            "mu": mu_hat,
            "sigma": heston_params["sigma"],
            "v0": heston_params["v0"],
            "theta": heston_params["theta"],
            "kappa": heston_params["kappa"],
            "rho": heston_params["rho"],
            "lambda_j": lambda_j,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
        }

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

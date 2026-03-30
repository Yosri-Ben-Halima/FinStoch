"""Heston stochastic volatility model."""

import warnings
from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess
from FinStoch.utils.plotting import plot_simulated_paths


@dataclass(kw_only=True)
class HestonModel(StochasticProcess):
    """Heston stochastic volatility model simulator.

    Models an asset price with stochastic variance:
        dS = mu * S * dt + sqrt(v) * S * dW_s
        dv = kappa * (theta - v) * dt + sigma * sqrt(v) * dW_v
        corr(dW_s, dW_v) = rho
    """

    v0: float
    theta: float
    kappa: float
    rho: float

    def simulate(self, seed: int | None = None, method: str = "euler") -> tuple[np.ndarray, np.ndarray]:
        """Simulate paths of the Heston model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' for Euler-Maruyama, 'milstein' for Milstein scheme.
            Milstein adds 0.25 * sigma^2 * (Wv^2 - 1) * dt to the
            variance process. 'exact' falls back to 'euler' with a
            warning (no closed-form path simulation).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple (S, v) of 2D arrays of shape (num_paths, num_steps)
            for asset prices and variance paths respectively.
        """
        self._validate_method(method)
        if method == "exact":
            warnings.warn(
                "Exact transition density is not available for the Heston model. Falling back to Euler-Maruyama.",
                stacklevel=2,
            )
            method = "euler"
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0

        v = np.zeros((self.num_paths, self._num_steps))
        v[:, 0] = self.v0

        L = np.array([[1, 0], [self.rho, np.sqrt(1 - self.rho**2)]])
        Xs_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))
        Xv_all = np.random.normal(0, 1, (self.num_paths, self._num_steps - 1))

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

            S[:, t] = S[:, t - 1] * np.exp(
                (self.mu - 0.5 * v[:, t - 1]) * self._dt + np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Ws
            )

        return S, v

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252, **kwargs: object) -> dict[str, float]:
        """Estimate Heston parameters from a price series.

        Stage 1: Compute rolling realized variance as a proxy for
        the latent variance process. Stage 2: Calibrate CIR dynamics
        to the realized variance series. Stage 3: Estimate mu, v0,
        and rho from return–variance relationships.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).
        rv_window : int, optional
            Rolling window for realized variance. Default is 21.

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'v0', 'theta',
            'kappa', 'rho'.

        References
        ----------
        Bollerslev, T. & Zhou, H. (2002). Estimating stochastic
        volatility diffusion using conditional moments of integrated
        volatility. Journal of Econometrics, 109(1), 33-65.
        """
        import pandas as pd

        from FinStoch.processes.cir import CoxIngersollRoss

        rv_window = int(kwargs.get("rv_window", 21))  # type: ignore[call-overload]

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < rv_window + 10:
            raise ValueError(f"data must be a 1D array with at least {rv_window + 10} observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        log_returns = np.diff(np.log(data))

        # Stage 1: Rolling realized variance
        rv = pd.Series(log_returns**2).rolling(rv_window).mean().dropna().values / dt
        rv = np.maximum(rv, 1e-10)

        # Stage 2: CIR calibration on realized variance
        cir_params = CoxIngersollRoss.calibrate(rv, dt=dt)
        kappa_hat = cir_params["theta"]  # CIR theta = Heston kappa
        theta_hat = cir_params["mu"]  # CIR mu = Heston theta
        sigma_hat = cir_params["sigma"]

        # Stage 3: v0, mu, rho
        v0_hat = float(rv[0])
        mu_hat = float(np.mean(log_returns) / dt + 0.5 * np.mean(rv))

        # Correlation: align return innovations with variance changes
        dv = np.diff(rv)
        r_aligned = log_returns[rv_window:]
        min_len = min(len(dv), len(r_aligned))
        if min_len > 2:
            rho_hat = float(np.corrcoef(r_aligned[:min_len], dv[:min_len])[0, 1])
            rho_hat = float(np.clip(rho_hat, -0.999, 0.999))
        else:
            rho_hat = 0.0

        return {
            "mu": mu_hat,
            "sigma": sigma_hat,
            "v0": v0_hat,
            "theta": theta_hat,
            "kappa": kappa_hat,
            "rho": rho_hat,
        }

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Heston Model",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Heston paths.

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

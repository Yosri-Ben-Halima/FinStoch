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
            'euler' and 'exact' are supported (both produce identical
            results using the time-changed Brownian motion
            representation). The Milstein scheme is not applicable.

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

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate VG parameters via the Method of Moments.

        Matches sample variance, skewness, and excess kurtosis of
        log-returns to the theoretical VG moment formulas and solves
        via numerical optimization.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'theta', 'nu'.

        References
        ----------
        Madan, D.B., Carr, P. & Chang, E.C. (1998). The Variance Gamma
        process and option pricing. European Finance Review, 2, 79-105.
        """
        import warnings

        from scipy import stats
        from scipy.optimize import minimize

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        log_returns = np.diff(np.log(data))
        m1 = float(np.mean(log_returns))
        m2 = float(np.var(log_returns, ddof=1))
        m3 = float(stats.skew(log_returns))
        m4 = float(stats.kurtosis(log_returns))  # excess kurtosis

        def moment_residuals(params: np.ndarray) -> float:
            theta_vg, sigma_vg, nu_vg = params
            var_th = (sigma_vg**2 + theta_vg**2 * nu_vg) * dt
            mu3_th = (2 * theta_vg**3 * nu_vg**2 + 3 * sigma_vg**2 * theta_vg * nu_vg) * dt
            mu4_excess_th = 3 * dt * nu_vg * (sigma_vg**4 + 2 * theta_vg**2 * sigma_vg**2 * nu_vg + 2 * theta_vg**4 * nu_vg**2)
            if var_th <= 0:
                return 1e30
            skew_th = mu3_th / var_th**1.5
            kurt_th = mu4_excess_th / var_th**2
            return float((m2 - var_th) ** 2 / max(m2**2, 1e-30) + (m3 - skew_th) ** 2 + (m4 - kurt_th) ** 2)

        sigma_init = np.sqrt(max(m2 / dt, 1e-10))
        nu_init = max(m4 / 3, 0.01)
        # Use skewness to set initial theta direction
        theta_init = m3 * (m2**1.5) / (3 * dt) if abs(m3) > 1e-6 else 0.0
        theta_init = np.clip(theta_init, -5, 5)

        bounds = [(-10, 10), (1e-6, 10), (1e-6, 10)]
        # Try multiple starting points to avoid local minima
        best_result = minimize(moment_residuals, x0=[0.0, sigma_init, 0.1], method="L-BFGS-B", bounds=bounds)
        best_val = best_result.fun
        for x0 in [
            [theta_init, sigma_init, nu_init],
            [0.0, sigma_init, 0.1],
            [-0.1, sigma_init * 0.8, 0.3],
        ]:
            res = minimize(moment_residuals, x0=x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_val:
                best_val = res.fun
                best_result = res

        theta_hat, sigma_hat, nu_hat = float(best_result.x[0]), float(best_result.x[1]), float(best_result.x[2])

        log_arg = 1 - theta_hat * nu_hat - 0.5 * sigma_hat**2 * nu_hat
        if log_arg <= 0:
            warnings.warn(
                "Estimated parameters violate the VG constraint "
                f"1 - theta*nu - 0.5*sigma^2*nu = {log_arg:.6f} <= 0. "
                "Results may be unreliable.",
                stacklevel=2,
            )
            omega_hat = 0.0
        else:
            omega_hat = (1 / nu_hat) * np.log(log_arg)

        mu_hat = float(m1 / dt - omega_hat - theta_hat)

        return {"mu": mu_hat, "sigma": sigma_hat, "theta": theta_hat, "nu": nu_hat}

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

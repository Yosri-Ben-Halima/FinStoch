"""Cox-Ingersoll-Ross process."""

import warnings
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

    def simulate(self, seed: int | None = None, method: str = "euler", antithetic: bool = False) -> np.ndarray:
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
            if antithetic:
                warnings.warn(
                    "Antithetic variates are not supported for CIR exact method. Ignoring.",
                    stacklevel=2,
                )
            exp_decay = np.exp(-self.theta * self._dt)
            c = self.sigma**2 * (1 - exp_decay) / (4 * self.theta)
            df = 4 * self.theta * self.mu / self.sigma**2
            for t in range(1, self._num_steps):
                nc = S[:, t - 1] * exp_decay / c
                S[:, t] = c * np.random.noncentral_chisquare(df, nc)
        else:
            Z_all = self._generate_normals((self.num_paths, self._num_steps - 1), antithetic)
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                drift = self.theta * (self.mu - S[:, t - 1]) * self._dt
                diffusion = self.sigma * np.sqrt(S[:, t - 1]) * np.sqrt(self._dt) * Z
                S[:, t] = S[:, t - 1] + drift + diffusion

                if method == "milstein":
                    S[:, t] += 0.25 * self.sigma**2 * (Z**2 - 1) * self._dt

                S[:, t] = np.maximum(S[:, t], 0)

        return S

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate CIR parameters via Conditional Least Squares.

        Stage 1: Estimate (theta, mu) by minimizing the sum of squared
        deviations from the exact conditional mean. Stage 2: Estimate
        sigma analytically from the conditional variance formula.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed non-negative values (chronological).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'theta'.

        References
        ----------
        Overbeck, L. & Rydberg, T. (1997). Estimation for diffusion
        processes from discrete observation.
        """
        from scipy.optimize import minimize

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data < 0):
            raise ValueError("data must be non-negative and contain no NaN values.")

        X = data[:-1]
        Y = data[1:]

        # Initial guess via OU-style OLS
        X_mean, Y_mean = np.mean(X), np.mean(Y)
        denom = np.sum((X - X_mean) ** 2)
        phi_init = np.clip(np.sum((X - X_mean) * (Y - Y_mean)) / denom, 1e-6, 1 - 1e-6)
        theta_init = max(-np.log(phi_init) / dt, 1e-6)
        mu_init = max(Y_mean - phi_init * X_mean, 1e-8) / (1 - phi_init)

        def cls_objective(params: np.ndarray) -> float:
            theta_val, mu_val = params
            exp_decay = np.exp(-theta_val * dt)
            cond_mean = X * exp_decay + mu_val * (1 - exp_decay)
            return float(np.sum((Y - cond_mean) ** 2))

        result = minimize(
            cls_objective,
            x0=[theta_init, mu_init],
            method="L-BFGS-B",
            bounds=[(1e-6, 100.0), (1e-8, None)],
        )
        theta_hat, mu_hat = float(result.x[0]), float(result.x[1])

        # Analytical sigma from conditional variance
        exp_decay = np.exp(-theta_hat * dt)
        cond_mean = X * exp_decay + mu_hat * (1 - exp_decay)
        sq_resid = (Y - cond_mean) ** 2
        A = X * (1 / theta_hat) * (exp_decay - exp_decay**2)
        B = mu_hat * (1 / (2 * theta_hat)) * (1 - exp_decay) ** 2
        sigma_sq = float(np.mean(sq_resid) / np.mean(A + B))
        sigma_hat = float(np.sqrt(max(sigma_sq, 1e-12)))

        if 2 * theta_hat * mu_hat < sigma_hat**2:
            warnings.warn(
                f"Feller condition violated: 2*theta*mu={2 * theta_hat * mu_hat:.6f} < sigma^2={sigma_hat**2:.6f}. "
                "The process may hit zero.",
                stacklevel=2,
            )

        return {"mu": mu_hat, "sigma": sigma_hat, "theta": theta_hat}

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

"""Ornstein-Uhlenbeck process."""

import warnings
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

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate OU parameters from observed data via AR(1) MLE.

        The exact OU transition is an AR(1) process. Parameters are
        recovered from OLS regression in closed form.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed values (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'theta'.

        References
        ----------
        Vasicek, O. (1977). An equilibrium characterization of the
        term structure. Journal of Financial Economics, 5(2), 177-188.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)):
            raise ValueError("data must not contain NaN values.")

        X = data[:-1]
        Y = data[1:]
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        phi_hat = float(np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2))

        if phi_hat <= 0 or phi_hat >= 1:
            warnings.warn(
                f"Estimated AR(1) coefficient phi={phi_hat:.4f} is outside (0, 1). "
                "Data may not be mean-reverting. Clamping to valid range.",
                stacklevel=2,
            )
            phi_hat = np.clip(phi_hat, 1e-6, 1 - 1e-6)

        c_hat = Y_mean - phi_hat * X_mean
        theta_hat = float(-np.log(phi_hat) / dt)
        mu_hat = float(c_hat / (1 - phi_hat))

        residuals = Y - c_hat - phi_hat * X
        var_eps = float(np.mean(residuals**2))
        sigma_hat = float(np.sqrt(var_eps * 2 * theta_hat / (1 - phi_hat**2)))

        return {"mu": mu_hat, "sigma": sigma_hat, "theta": theta_hat}

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

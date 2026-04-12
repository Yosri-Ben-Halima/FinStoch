"""Vasicek interest rate model."""

import warnings
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

    def simulate(self, seed: int | None = None, method: str = "euler", antithetic: bool = False) -> np.ndarray:
        """Simulate paths of the Vasicek model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' or 'milstein'. Both produce identical results for
            Vasicek since the diffusion is constant (dg/dr = 0).
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
        Z_all = self._generate_normals((self.num_paths, self._num_steps - 1), antithetic)

        if method == "exact":
            decay = np.exp(-self.a * self._dt)
            mean_weight = 1 - decay
            std_exact = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * self._dt)) / (2 * self.a))
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                S[:, t] = S[:, t - 1] * decay + self.mu * mean_weight + std_exact * Z
        else:
            for t in range(1, self._num_steps):
                Z = Z_all[:, t - 1]
                drift = self.a * (self.mu - S[:, t - 1]) * self._dt
                diffusion = self.sigma * np.sqrt(self._dt) * Z
                S[:, t] = S[:, t - 1] + drift + diffusion
                # Milstein correction is zero (constant diffusion, dg/dr = 0)

        return S

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
        """Estimate Vasicek parameters from observed rates via AR(1) MLE.

        Mathematically identical to OU calibration with parameter ``a``
        (mean reversion speed) in place of ``theta``.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed interest rates (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'a'.

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
        a_hat = float(-np.log(phi_hat) / dt)
        mu_hat = float(c_hat / (1 - phi_hat))

        residuals = Y - c_hat - phi_hat * X
        var_eps = float(np.mean(residuals**2))
        sigma_hat = float(np.sqrt(var_eps * 2 * a_hat / (1 - phi_hat**2)))

        return {"mu": mu_hat, "sigma": sigma_hat, "a": a_hat}

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

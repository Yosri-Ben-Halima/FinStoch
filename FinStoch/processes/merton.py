"""Merton Jump Diffusion process."""

from dataclasses import dataclass

import numpy as np

from FinStoch.processes.base import StochasticProcess


@dataclass(kw_only=True)
class MertonJumpDiffusion(StochasticProcess):
    """Merton Jump Diffusion process simulator.

    Extends GBM by incorporating Poisson-distributed jumps:
        dS = (mu - lambda_j * k) * S * dt + sigma * S * dW + J * S * dN
    """

    lambda_j: float
    mu_j: float
    sigma_j: float

    def simulate(self, seed: int | None = None, method: str = "euler", antithetic: bool = False) -> np.ndarray:
        """Simulate paths of the Merton Jump Diffusion model.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            'euler' uses the exact log-normal diffusion (default).
            'milstein' uses Euler-Milstein for the diffusion component;
            jumps remain multiplicative and are unaffected.
            'exact' is accepted as an alias for 'euler' (both use the
            exact log-normal diffusion with exact Poisson jumps).

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

        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

        Z_all = self._generate_normals((self.num_paths, self._num_steps - 1), antithetic)
        N_all = np.random.poisson(self.lambda_j * self._dt, (self.num_paths, self._num_steps - 1))
        J_all = np.random.normal(self.mu_j, self.sigma_j, (self.num_paths, self._num_steps - 1))

        for t in range(1, self._num_steps):
            Z = Z_all[:, t - 1]
            N = N_all[:, t - 1]
            J = np.where(N > 0, J_all[:, t - 1], 0.0)

            if method == "milstein":
                S[:, t] = (
                    S[:, t - 1]
                    + (self.mu - self.lambda_j * k) * S[:, t - 1] * self._dt
                    + self.sigma * S[:, t - 1] * np.sqrt(self._dt) * Z
                    + 0.5 * self.sigma**2 * S[:, t - 1] * (Z**2 - 1) * self._dt
                    + S[:, t - 1] * (np.exp(J) - 1)
                )
            else:
                S[:, t] = S[:, t - 1] * np.exp(
                    (self.mu - 0.5 * self.sigma**2 - self.lambda_j * k) * self._dt + self.sigma * np.sqrt(self._dt) * Z + J
                )

        return S

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252, **kwargs: object) -> dict[str, float]:
        """Estimate Merton parameters via the EM algorithm.

        Treats jump indicators as latent variables. E-step computes
        posterior jump probability per observation; M-step updates all
        parameters in closed form. Fully vectorized.

        Parameters
        ----------
        data : np.ndarray
            1D array of observed prices (chronological order).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).
        max_iter : int, optional
            Maximum EM iterations. Default is 200.
        tol : float, optional
            Log-likelihood convergence tolerance. Default is 1e-8.

        Returns
        -------
        dict[str, float]
            Estimated parameters: 'mu', 'sigma', 'lambda_j', 'mu_j',
            'sigma_j'.

        References
        ----------
        Honore, P. (1998). Pitfalls in estimating jump-diffusion models.
        """
        from scipy.stats import norm

        max_iter = int(kwargs.get("max_iter", 200))  # type: ignore[call-overload]
        tol = float(kwargs.get("tol", 1e-8))  # type: ignore[call-overload,arg-type]

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or len(data) < 3:
            raise ValueError("data must be a 1D array with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        log_returns = np.diff(np.log(data))
        n = len(log_returns)

        # Initialize
        sigma = float(np.std(log_returns, ddof=1) / np.sqrt(dt))
        mu = float(np.mean(log_returns) / dt + 0.5 * sigma**2)
        lambda_j = 0.1
        mu_j = 0.0
        sigma_j = max(2 * sigma * np.sqrt(dt), 1e-6)

        prev_ll = -np.inf
        for _ in range(max_iter):
            k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
            mean_nj = (mu - 0.5 * sigma**2 - lambda_j * k) * dt
            std_nj = sigma * np.sqrt(dt)

            mean_j = mean_nj + mu_j
            std_j = np.sqrt(std_nj**2 + sigma_j**2)

            # E-step
            p_nj = np.maximum((1 - lambda_j * dt), 1e-300) * norm.pdf(log_returns, mean_nj, std_nj)
            p_j = np.maximum(lambda_j * dt, 1e-300) * norm.pdf(log_returns, mean_j, std_j)
            total = p_nj + p_j + 1e-300
            tau = p_j / total

            ll = float(np.sum(np.log(total)))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # M-step
            tau_sum = np.sum(tau)
            lambda_j = max(float(tau_sum / (n * dt)), 1e-10)

            # Jump component
            if tau_sum > 1e-10:
                w_j = tau / tau_sum
                mu_j = float(np.sum(w_j * (log_returns - mean_nj)))
                sigma_j = max(float(np.sqrt(np.sum(w_j * (log_returns - mean_nj - mu_j) ** 2))), 1e-10)
            else:
                mu_j = 0.0
                sigma_j = 1e-6

            # Diffusion component
            nj_sum = np.sum(1 - tau)
            if nj_sum > 1e-10:
                w_nj = (1 - tau) / nj_sum
                mean_r_nj = float(np.sum(w_nj * log_returns))
                sigma = max(float(np.sqrt(np.sum(w_nj * (log_returns - mean_r_nj) ** 2) / dt)), 1e-10)
                k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
                mu = float(mean_r_nj / dt + 0.5 * sigma**2 + lambda_j * k)

        return {
            "mu": mu,
            "sigma": sigma,
            "lambda_j": lambda_j,
            "mu_j": mu_j,
            "sigma_j": sigma_j,
        }

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Merton Model",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated Merton Jump Diffusion paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

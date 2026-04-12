"""Correlated multi-asset Geometric Brownian Motion simulator."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from scipy.linalg import cholesky

from FinStoch.utils.timesteps import (
    generate_date_range_with_granularity,
    date_range_duration,
)


@dataclass(kw_only=True)
class CorrelatedGBM:
    """Multi-asset correlated Geometric Brownian Motion simulator.

    Models N correlated assets, each following GBM:
        dS_i = mu_i * S_i * dt + sigma_i * S_i * dW_i
        corr(dW_i, dW_j) = rho_ij

    Correlation is implemented via Cholesky decomposition of the
    correlation matrix applied to independent standard normals.

    Parameters
    ----------
    S0 : list[float]
        Initial prices for each asset.
    mu : list[float]
        Drift coefficients for each asset.
    sigma : list[float]
        Volatility coefficients for each asset.
    correlation : list[list[float]]
        Correlation matrix (num_assets x num_assets).
    num_paths : int
        Number of paths to simulate per asset.
    start_date : str
        Start date for the simulation.
    end_date : str
        End date for the simulation.
    granularity : str
        Time granularity for each step.
    business_days : bool, optional
        Use business days when granularity is 'D'. Default is False.
    """

    S0: list[float]
    mu: list[float]
    sigma: list[float]
    correlation: list[list[float]]
    num_paths: int
    start_date: str
    end_date: str
    granularity: str
    business_days: bool = False

    def __post_init__(self) -> None:
        self._S0 = np.asarray(self.S0, dtype=np.float64)
        self._mu = np.asarray(self.mu, dtype=np.float64)
        self._sigma = np.asarray(self.sigma, dtype=np.float64)
        self._corr = np.asarray(self.correlation, dtype=np.float64)

        self._num_assets = len(self._S0)

        if not (len(self._mu) == len(self._sigma) == self._num_assets):
            raise ValueError("S0, mu, and sigma must have the same length.")
        if self._corr.shape != (self._num_assets, self._num_assets):
            raise ValueError(f"Correlation matrix must be ({self._num_assets}, {self._num_assets}).")
        if not np.allclose(self._corr, self._corr.T):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(self._corr), 1.0):
            raise ValueError("Correlation matrix diagonal must be all ones.")
        eigvals = np.linalg.eigvalsh(self._corr)
        if np.any(eigvals < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite.")

        self._L = cholesky(self._corr + np.eye(self._num_assets) * 1e-12, lower=True)
        self._recalculate_time_grid()

    def _recalculate_time_grid(self) -> None:
        self._t = generate_date_range_with_granularity(self.start_date, self.end_date, self.granularity, self.business_days)
        self._T = date_range_duration(self._t)
        self._num_steps = len(self._t)
        self._dt = self._T / self._num_steps

    # --- Computed properties ---

    @property
    def num_assets(self) -> int:
        return self._num_assets

    @property
    def T(self) -> float:
        return self._T

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def t(self) -> DatetimeIndex:
        return self._t

    # --- Simulation ---

    def simulate(self, seed: int | None = None, antithetic: bool = False) -> np.ndarray:
        """Simulate correlated multi-asset GBM paths.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool, optional
            If True, use antithetic variates. Default is False.

        Returns
        -------
        np.ndarray
            3D array of shape (num_assets, num_paths, num_steps).
        """
        if seed is not None:
            np.random.seed(seed)

        n_a = self._num_assets
        n_p = self.num_paths
        n_s = self._num_steps
        dt = self._dt

        # Generate independent normals
        if antithetic:
            half = (n_p + 1) // 2
            Z_half = np.random.normal(0, 1, (n_a, half, n_s - 1))
            Z_ind = np.concatenate([Z_half, -Z_half], axis=1)[:, :n_p, :]
        else:
            Z_ind = np.random.normal(0, 1, (n_a, n_p, n_s - 1))

        # Correlate via Cholesky: L @ Z_ind across asset dimension
        Z_corr = np.einsum("ab,bpt->apt", self._L, Z_ind)

        # Fully vectorized GBM (exact log-normal, no time loop)
        drift = (self._mu - 0.5 * self._sigma**2)[:, np.newaxis, np.newaxis] * dt
        diffusion = self._sigma[:, np.newaxis, np.newaxis] * np.sqrt(dt) * Z_corr
        log_increments = drift + diffusion

        S = np.zeros((n_a, n_p, n_s))
        S[:, :, 0] = self._S0[:, np.newaxis]
        cum_log = np.cumsum(log_increments, axis=2)
        S[:, :, 1:] = self._S0[:, np.newaxis, np.newaxis] * np.exp(cum_log)

        return S

    # --- Analytics ---

    def expected_path(self, paths: np.ndarray, asset: int | None = None) -> np.ndarray:
        """Mean path across simulations.

        Parameters
        ----------
        paths : np.ndarray
            (num_assets, num_paths, num_steps) simulation output.
        asset : int, optional
            If given, return mean for that asset only (1D).
            If None, return (num_assets, num_steps).
        """
        if asset is not None:
            return np.mean(paths[asset], axis=0)
        return np.mean(paths, axis=1)

    def portfolio_paths(self, paths: np.ndarray, weights: list[float]) -> np.ndarray:
        """Compute weighted portfolio value paths.

        Parameters
        ----------
        paths : np.ndarray
            (num_assets, num_paths, num_steps) simulation output.
        weights : list[float]
            Portfolio weights per asset.

        Returns
        -------
        np.ndarray
            2D array of shape (num_paths, num_steps).
        """
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != self._num_assets:
            raise ValueError(f"weights must have length {self._num_assets}.")
        return np.einsum("a,apt->pt", w, paths)

    def realized_correlation(self, paths: np.ndarray) -> np.ndarray:
        """Compute realized correlation from simulated log-returns.

        Returns
        -------
        np.ndarray
            (num_assets, num_assets) realized correlation matrix.
        """
        log_ret = np.diff(np.log(paths), axis=2)
        flat = log_ret.reshape(self._num_assets, -1)
        return np.corrcoef(flat)

    def var(self, portfolio_paths: np.ndarray, alpha: float = 0.05) -> float:
        """Value at Risk on portfolio terminal values."""
        return float(np.percentile(portfolio_paths[:, -1], 100 * alpha))

    def cvar(self, portfolio_paths: np.ndarray, alpha: float = 0.05) -> float:
        """Conditional Value at Risk on portfolio terminal values."""
        terminal = portfolio_paths[:, -1]
        threshold = np.percentile(terminal, 100 * alpha)
        return float(np.mean(terminal[terminal <= threshold]))

    def max_drawdown(self, portfolio_paths: np.ndarray) -> np.ndarray:
        """Maximum drawdown per portfolio path."""
        cummax = np.maximum.accumulate(portfolio_paths, axis=1)
        drawdowns = (cummax - portfolio_paths) / cummax
        return np.max(drawdowns, axis=1)

    # --- Data conversion ---

    def to_dataframe(self, paths: np.ndarray, asset: int = 0) -> pd.DataFrame:
        """Convert single-asset paths to a DataFrame.

        Parameters
        ----------
        paths : np.ndarray
            (num_assets, num_paths, num_steps) simulation output.
        asset : int
            Asset index to extract. Default is 0.
        """
        return pd.DataFrame(paths[asset], columns=self._t)

    # --- Calibration ---

    @classmethod
    def calibrate(cls, data: np.ndarray, dt: float = 1 / 252) -> dict:
        """Calibrate from multi-asset price data.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (num_observations, num_assets).
        dt : float, optional
            Time step between observations. Default is 1/252 (daily).

        Returns
        -------
        dict
            Keys: 'S0', 'mu', 'sigma', 'correlation'.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 3:
            raise ValueError("data must be 2D with at least 3 observations.")
        if np.any(np.isnan(data)) or np.any(data <= 0):
            raise ValueError("data must be positive and contain no NaN values.")

        log_returns = np.diff(np.log(data), axis=0)
        sigma_arr = np.std(log_returns, axis=0, ddof=1) / np.sqrt(dt)
        mu_arr = np.mean(log_returns, axis=0) / dt + 0.5 * sigma_arr**2

        return {
            "S0": data[-1].tolist(),
            "mu": mu_arr.tolist(),
            "sigma": sigma_arr.tolist(),
            "correlation": np.corrcoef(log_returns.T).tolist(),
        }

    # --- Plotting ---

    def plot(
        self,
        paths: np.ndarray | None = None,
        asset: int | None = None,
        title: str = "Correlated GBM",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
    ) -> None:
        """Plot simulated paths for one or all assets."""
        if paths is None:
            paths = self.simulate()

        if asset is not None:
            plt.figure(figsize=fig_size)
            plt.plot(self._t, paths[asset].T, alpha=0.6)
            plt.title(f"{title} — Asset {asset}")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()
        else:
            fig, axes = plt.subplots(
                self._num_assets,
                1,
                figsize=fig_size or (10, 3 * self._num_assets),
                sharex=True,
            )
            if self._num_assets == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                ax.plot(self._t, paths[i].T, alpha=0.6)
                ax.set_title(f"Asset {i}")
                ax.set_ylabel(ylabel)
                ax.grid(True)
            axes[-1].set_xlabel("Time")  # type: ignore[union-attr]
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

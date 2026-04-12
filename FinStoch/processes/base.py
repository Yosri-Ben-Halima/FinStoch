"""Base class for all stochastic process simulators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from scipy import stats

from FinStoch.utils.plotting import plot_simulated_paths
from FinStoch.utils.timesteps import (
    generate_date_range_with_granularity,
    date_range_duration,
)


@dataclass(kw_only=True)
class StochasticProcess(ABC):
    """Abstract base class for stochastic process simulators.

    Provides shared initialization, time grid management, plotting,
    data conversion, and analytics for all stochastic processes.

    Parameters
    ----------
    S0 : float
        The initial value of the process.
    mu : float
        The drift coefficient.
    sigma : float
        The volatility coefficient.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation (e.g., '2023-09-01').
    end_date : str
        The end date for the simulation (e.g., '2023-12-31').
    granularity : str
        The time granularity for each step (e.g., 'D', 'H', '10T').
    business_days : bool, optional
        If True, use business days instead of calendar days when
        granularity is 'D'. Default is False.
    """

    S0: float
    mu: float
    sigma: float
    num_paths: int
    start_date: str
    end_date: str
    granularity: str
    business_days: bool = False

    _VALID_METHODS: ClassVar[tuple[str, ...]] = ("euler", "milstein", "exact")
    _TIME_FIELDS: ClassVar[frozenset[str]] = frozenset({"start_date", "end_date", "granularity", "business_days"})

    def __post_init__(self) -> None:
        self._recalculate_time_grid()

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name in self._TIME_FIELDS and hasattr(self, "_dt"):
            self._recalculate_time_grid()

    def _recalculate_time_grid(self) -> None:
        """Recompute time grid attributes from date range and granularity."""
        self._t = generate_date_range_with_granularity(self.start_date, self.end_date, self.granularity, self.business_days)
        self._T = date_range_duration(self._t)
        self._num_steps = len(self._t)
        self._dt = self._T / self._num_steps

    def _validate_method(self, method: str) -> None:
        """Validate the discretization method parameter.

        Raises
        ------
        ValueError
            If method is not 'euler', 'milstein', or 'exact'.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Must be one of {self._VALID_METHODS}.")

    def _generate_normals(self, shape: tuple[int, int], antithetic: bool) -> np.ndarray:
        """Generate standard normal draws, optionally with antithetic mirroring.

        Parameters
        ----------
        shape : tuple[int, int]
            (num_paths, num_steps - 1).
        antithetic : bool
            If True, generate half the draws and mirror them.

        Returns
        -------
        np.ndarray
            Array of shape ``shape``.
        """
        if not antithetic:
            return np.random.normal(0, 1, shape)
        half = (shape[0] + 1) // 2
        Z_half = np.random.normal(0, 1, (half, shape[1]))
        return np.concatenate([Z_half, -Z_half], axis=0)[: shape[0]]

    @abstractmethod
    def simulate(
        self,
        seed: int | None = None,
        method: str = "euler",
        antithetic: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Simulate paths of the stochastic process.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. If None, no seed is set.
        method : str, optional
            Discretization scheme. 'euler' for Euler-Maruyama (default),
            'milstein' for the Milstein scheme (higher-order accuracy),
            'exact' for exact transition density sampling (where available).
            Not all processes support all methods; see per-class docs.
        antithetic : bool, optional
            If True, use antithetic variates for variance reduction.
            Generates num_paths // 2 independent draws and mirrors them.
            Output shape is unchanged. Default is False.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            A 2D array of shape (num_paths, num_steps), or a tuple of two
            such arrays for models with multiple outputs (e.g., Heston).
        """
        ...

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Simulated Paths",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated paths.

        Parameters
        ----------
        paths : np.ndarray, optional
            Pre-computed paths to plot. If None, calls simulate().
        title : str
            Plot title.
        ylabel : str
            Y-axis label.
        fig_size : tuple, optional
            Figure size in inches.
        **kwargs
            Additional keyword arguments passed to plot_simulated_paths.
        """
        plot_simulated_paths(
            self._t,
            self.simulate,
            paths,
            title=title,
            ylabel=ylabel,
            fig_size=fig_size,
            grid=kwargs.get("grid", True),
        )

    # --- Computed properties (read-only) ---

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

    # --- Data conversion ---

    def to_dataframe(
        self,
        paths: np.ndarray | tuple[np.ndarray, np.ndarray],
        variance: bool = False,
    ) -> pd.DataFrame:
        """Convert simulation output to a pandas DataFrame.

        Parameters
        ----------
        paths : np.ndarray or tuple[np.ndarray, np.ndarray]
            Simulation output from simulate(). For Heston-style tuple
            output, use ``variance=True`` to select the variance array.
        variance : bool, optional
            If True and paths is a tuple, use the second (variance) array.
            Default is False (use the first / price array).

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex as columns and path indices as rows.
        """
        if isinstance(paths, tuple):
            data = paths[1] if variance else paths[0]
        else:
            data = paths
        return pd.DataFrame(data, columns=self._t)

    # --- Analytics ---

    def summary_statistics(self, paths: np.ndarray) -> dict[str, np.ndarray]:
        """Compute summary statistics across paths at each time step.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with keys 'mean', 'std', 'skew', 'kurtosis',
            'min', 'max', each mapping to a 1D array of length num_steps.
        """
        return {
            "mean": np.mean(paths, axis=0),
            "std": np.std(paths, axis=0),
            "skew": stats.skew(paths, axis=0),
            "kurtosis": stats.kurtosis(paths, axis=0),
            "min": np.min(paths, axis=0),
            "max": np.max(paths, axis=0),
        }

    def terminal_distribution(
        self,
        paths: np.ndarray,
        bins: int = 50,
        fig_size: tuple | None = None,
    ) -> None:
        """Plot histogram of terminal values with a fitted normal overlay.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).
        bins : int, optional
            Number of histogram bins. Default is 50.
        fig_size : tuple, optional
            Figure size in inches.
        """
        terminal = paths[:, -1]
        mu_fit, std_fit = stats.norm.fit(terminal)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.hist(terminal, bins=bins, density=True, alpha=0.7, label="Terminal values")

        x = np.linspace(terminal.min(), terminal.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu_fit, std_fit), "r-", lw=2, label="Fitted normal")

        ax.set_title("Terminal Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)
        plt.show()

    def confidence_bands(self, paths: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute percentile-based confidence bands across paths.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).
        level : float, optional
            Confidence level between 0 and 1. Default is 0.95.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper confidence band arrays, each of length num_steps.
        """
        alpha = 1 - level
        lower = np.percentile(paths, 100 * alpha / 2, axis=0)
        upper = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def expected_path(self, paths: np.ndarray) -> np.ndarray:
        """Compute the mean path across all simulated paths.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).

        Returns
        -------
        np.ndarray
            1D array of length num_steps representing the mean path.
        """
        return np.mean(paths, axis=0)

    def median_path(self, paths: np.ndarray) -> np.ndarray:
        """Compute the median path across all simulated paths.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).

        Returns
        -------
        np.ndarray
            1D array of length num_steps representing the median path.
        """
        return np.median(paths, axis=0)

    def var(self, paths: np.ndarray, alpha: float = 0.05) -> float:
        """Compute Value at Risk at the terminal time step.

        VaR is the alpha-quantile of terminal values, representing the
        worst expected loss at the given confidence level.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).
        alpha : float, optional
            Significance level. Default is 0.05 (95% VaR).

        Returns
        -------
        float
            The Value at Risk.
        """
        terminal = paths[:, -1]
        return float(np.percentile(terminal, 100 * alpha))

    def cvar(self, paths: np.ndarray, alpha: float = 0.05) -> float:
        """Compute Conditional Value at Risk (Expected Shortfall) at terminal.

        CVaR is the expected value of terminal prices that fall below the
        VaR threshold.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).
        alpha : float, optional
            Significance level. Default is 0.05 (95% CVaR).

        Returns
        -------
        float
            The Conditional Value at Risk.
        """
        terminal = paths[:, -1]
        var_threshold = np.percentile(terminal, 100 * alpha)
        return float(np.mean(terminal[terminal <= var_threshold]))

    def max_drawdown(self, paths: np.ndarray) -> np.ndarray:
        """Compute maximum drawdown for each simulated path.

        Maximum drawdown is the largest peak-to-trough decline observed
        along a path, expressed as a fraction of the peak value.

        Parameters
        ----------
        paths : np.ndarray
            A 2D array of shape (num_paths, num_steps).

        Returns
        -------
        np.ndarray
            1D array of length num_paths with the maximum drawdown for
            each path (values between 0 and 1).
        """
        cummax = np.maximum.accumulate(paths, axis=1)
        drawdowns = (cummax - paths) / cummax
        return np.max(drawdowns, axis=1)

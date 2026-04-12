"""Bootstrap Monte Carlo simulation from historical data."""

import warnings

import numpy as np

from FinStoch.processes.base import StochasticProcess


class BootstrapMonteCarlo(StochasticProcess):
    """Bootstrap Monte Carlo simulator.

    Generates simulated price paths by resampling (with replacement)
    log returns computed from historical price data. No parametric
    model is assumed — the empirical return distribution is used
    directly.

    Parameters
    ----------
    historical_prices : np.ndarray
        1D array of historical prices (ordered chronologically).
        Log returns are computed internally.
    num_paths : int
        The number of paths to simulate.
    start_date : str
        The start date for the simulation.
    end_date : str
        The end date for the simulation.
    granularity : str
        The time granularity for each step.
    S0 : float, optional
        The initial price for simulated paths. Defaults to the last
        historical price.
    block_size : int, optional
        If greater than 1, uses block bootstrap to preserve
        autocorrelation structure. Default is 1 (i.i.d. bootstrap).
    business_days : bool, optional
        If True, use business days instead of calendar days when
        granularity is 'D'. Default is False.
    """

    def __init__(
        self,
        historical_prices: np.ndarray,
        num_paths: int,
        start_date: str,
        end_date: str,
        granularity: str,
        S0: float | None = None,
        block_size: int = 1,
        business_days: bool = False,
    ) -> None:
        prices = np.asarray(historical_prices, dtype=float)
        self._log_returns = np.diff(np.log(prices))
        self._block_size = block_size

        initial = float(prices[-1]) if S0 is None else S0
        mu = float(np.mean(self._log_returns))
        sigma = float(np.std(self._log_returns))

        super().__init__(
            S0=initial,
            mu=mu,
            sigma=sigma,
            num_paths=num_paths,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            business_days=business_days,
        )

    def simulate(self, seed: int | None = None, method: str = "euler", antithetic: bool = False) -> np.ndarray:
        """Simulate paths by bootstrapping historical log returns.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        method : str, optional
            Only 'euler' is accepted. The bootstrap does not use a
            discretization scheme, so 'milstein' raises ValueError
            and 'exact' falls back to 'euler' with a warning.

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
            raise ValueError("Milstein scheme is not applicable to Bootstrap Monte Carlo (no underlying SDE).")
        if method == "exact":
            warnings.warn(
                "Exact transition density is not applicable to Bootstrap Monte Carlo. Falling back to Euler-Maruyama.",
                stacklevel=2,
            )
            method = "euler"
        if antithetic:
            warnings.warn(
                "Antithetic variates are not applicable to Bootstrap Monte Carlo. Ignoring.",
                stacklevel=2,
            )
        if seed is not None:
            np.random.seed(seed)

        n_returns = self._num_steps - 1
        S = np.zeros((self.num_paths, self._num_steps))
        S[:, 0] = self.S0

        if self._block_size <= 1:
            idx = np.random.randint(0, len(self._log_returns), (self.num_paths, n_returns))
            sampled_returns = self._log_returns[idx]
        else:
            sampled_returns = self._block_bootstrap(n_returns)

        cumulative = np.cumsum(sampled_returns, axis=1)
        S[:, 1:] = self.S0 * np.exp(cumulative)

        return S

    def _block_bootstrap(self, n_returns: int) -> np.ndarray:
        """Resample log returns in contiguous blocks."""
        n_hist = len(self._log_returns)
        bs = self._block_size
        n_blocks = (n_returns + bs - 1) // bs

        max_start = n_hist - bs
        if max_start < 1:
            max_start = 1

        starts = np.random.randint(0, max_start, (self.num_paths, n_blocks))
        blocks = np.concatenate(
            [
                self._log_returns[starts[:, i] : starts[:, i] + bs]  # noqa: E203
                if self.num_paths == 1
                else np.column_stack([self._log_returns[s : s + bs] for s in starts[:, i]]).T
                for i in range(n_blocks)
            ],
            axis=1,
        )

        return blocks[:, :n_returns]

    def plot(
        self,
        paths: np.ndarray | None = None,
        title: str = "Bootstrap Monte Carlo",
        ylabel: str = "Value",
        fig_size: tuple | None = None,
        **kwargs: object,
    ) -> None:
        """Plot simulated bootstrap paths."""
        super().plot(paths, title=title, ylabel=ylabel, fig_size=fig_size, **kwargs)

    @property
    def log_returns(self) -> np.ndarray:
        """The historical log returns used for resampling."""
        return self._log_returns

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int) -> None:
        self._block_size = value

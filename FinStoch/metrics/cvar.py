import numpy as np
from .var import ValueAtRisk
from typing import Optional


class ExpectedShortfall:
    """A class to compute the Monte-Carlo Conditional Value at Risk (CVaR) or Expected Shortfall (ES)."""

    def __init__(self, simulated_paths: np.ndarray) -> None:
        """
        Initialize the Expected Shortfall class with simulated paths.

        Parameters
        ----------
        simulated_paths : np.ndarray
            The simulated paths of asset values.
        """
        self.simulated_paths = simulated_paths

    def calculate(
        self, alpha: Optional[float] = None, confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate the Monte-Carlo Conditional Value at Risk (CVaR) or Expected Shortfall (ES) at a given confidence level.

        Parameters
        ----------
        The significance level for ES or CVaR (e.g., 0.05 for 95% confidence). If provided,
            overrides confidence_level.
        confidence_level : float, optional
            The confidence level for ES or CVaR (e.g., 0.95 for 95% confidence).

        Returns
        -------
        float
            The calculated ES.
        """
        assert (confidence_level is None) != (
            alpha is None
        ), "Either 'alpha' or 'confidence_level' must be provided."
        # Calculate returns
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        # Calculate VaR
        var_calculator = ValueAtRisk(self.simulated_paths)
        var_threshold = var_calculator.calculate(
            confidence_level=confidence_level, alpha=alpha
        )

        # Calculate ES
        losses_exceeding_var = returns[returns <= var_threshold]
        es_value = (
            np.mean(losses_exceeding_var) if losses_exceeding_var.size > 0 else np.nan
        )
        return es_value

import numpy as np
from typing import Optional


class ValueAtRisk:
    """
    A class to compute the Monte-Carlo Value at Risk (VaR).
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        """
        Initialize the Value at Risk class with simulated paths.

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
        Calculate the Monte-Carlo Value at Risk (VaR) at a given confidence level or alpha.

        Parameters
        ----------
        alpha : float, optional
            The significance level for VaR (e.g., 0.05 for 95% confidence). If provided,
            overrides confidence_level.
        confidence_level : float, optional
            The confidence level for VaR (e.g., 0.95 for 95% confidence).

        Returns
        -------
        float
            The calculated VaR.

        Raises
        ------
        ValueError
            If neither `alpha` nor `confidence_level` is provided.
        """
        assert (alpha is None) != (
            confidence_level is None
        ), "Either 'alpha' or 'confidence_level' must be provided."

        if alpha is not None:
            confidence_level = 1 - alpha

        # Calculate returns
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        # Calculate VaR
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        return var_threshold

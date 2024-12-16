import numpy as np
from .returns import ExpectedReturn
from .vol import Volatility


class SortinoRatio:
    """
    A class to compute the Sortino Ratio.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self, risk_free_rate: float) -> float:
        """
        Calculate the Sortino Ratio.

        Parameters
        ----------
        risk_free_rate : float
            The risk free rate to calculate the Sortino Ratio.
        Returns
        -------
        float
            The Sortino Ratio for the given risk free rate.
        """
        downside_risk = Volatility(self.simulated_paths).downside_vol()
        expected_return = ExpectedReturn(self.simulated_paths).calculate()
        return (expected_return - risk_free_rate) / downside_risk

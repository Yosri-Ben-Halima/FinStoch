import numpy as np
from .returns import ExpectedReturn
from .vol import Volatility


class SharpeRatio:
    """
    A class to compute the Sharpe Ratio.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self, risk_free_rate: float) -> float:
        """
        Calculate the Sharpe Ratio.

        Parameters
        ----------
        risk_free_rate : float
            The risk free rate to calculate the Sharpe Ratio.
        Returns
        -------
        float
            The Sharpe Ratio for the given risk free rate.
        """
        expected_return = ExpectedReturn(self.simulated_paths).calculate()
        volatility = Volatility(self.simulated_paths).calculate()
        return (expected_return - risk_free_rate) / volatility

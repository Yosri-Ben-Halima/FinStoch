import numpy as np
from .var import ValueAtRisk

class ExpectedShortfall:
    def __init__(self, simulated_paths: np.ndarray):
        """
        Initialize the Expected Shortfall class with simulated paths.
        
        Parameters
        ----------
        simulated_paths : np.ndarray
            The simulated paths of asset values.
        """
        self.simulated_paths = simulated_paths

    def calculate(self, confidence_level: float) -> float:
        """
        Calculate Expected Shortfall (ES) at a given confidence level.
        
        Parameters
        ----------
        confidence_level : float
            The confidence level for ES (e.g., 0.95 for 95% confidence).
        
        Returns
        -------
        float
            The calculated ES.
        """
        # Calculate returns
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        # Calculate VaR
        var_calculator = ValueAtRisk(self.simulated_paths)
        var_threshold = var_calculator.calculate(confidence_level)
        # Calculate ES
        losses_exceeding_var = returns[returns <= var_threshold]
        es_value = np.mean(losses_exceeding_var) if losses_exceeding_var.size > 0 else np.nan
        return es_value

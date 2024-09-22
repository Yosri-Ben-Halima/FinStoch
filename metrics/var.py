import numpy as np

class ValueAtRisk:
    def __init__(self, simulated_paths: np.ndarray):
        """
        Initialize the Value at Risk class with simulated paths.
        
        Parameters
        ----------
        simulated_paths : np.ndarray
            The simulated paths of asset values.
        """
        self.simulated_paths = simulated_paths

    def calculate(self, confidence_level: float) -> float:
        """
        Calculate Value at Risk (VaR) at a given confidence level.
        
        Parameters
        ----------
        confidence_level : float
            The confidence level for VaR (e.g., 0.95 for 95% confidence).
        
        Returns
        -------
        float
            The calculated VaR.
        """
        # Calculate returns
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        # Calculate VaR
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        return var_threshold

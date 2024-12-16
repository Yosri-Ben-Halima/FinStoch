import numpy as np
from scipy.stats import entropy as scipy_entropy


class Entropy:
    """
    A class to compute the entropy of the return distribution.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self) -> float:
        """
        Calculate the entropy of the returns distribution to quantify the randomness and unpredictibility.
        """
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        histogram, _ = np.histogram(returns, bins=50, density=True)
        return scipy_entropy(histogram)

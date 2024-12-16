import numpy as np


class MaxDrawdown:
    """
    A class to compute the maximum drawdown from simulated paths.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self) -> float:
        """
        Calculates the maximum drawdown from the given simulated paths.

        Returns
        -------
        float:
            The maximum drawdown.
        """
        cumulative_max = np.maximum.accumulate(self.simulated_paths, axis=1)
        drawdowns = (self.simulated_paths - cumulative_max) / cumulative_max
        return np.min(drawdowns)

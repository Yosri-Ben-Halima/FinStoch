import numpy as np


class Kurtosis:
    """
    A class to compute the kurtosis of returns.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self) -> float:
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)

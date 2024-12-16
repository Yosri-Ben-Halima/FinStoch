import numpy as np


class Volatility:
    """
    A class to compute the volatility (standard deviation) of returns.
    """

    def __init__(self, simulated_paths: np.ndarray) -> None:
        self.simulated_paths = simulated_paths

    def calculate(self) -> float:
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        return np.std(returns)

    def downside_vol(self) -> float:
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        downside_returns = returns[returns < 0]
        return np.std(downside_returns)

    def upside_vol(self) -> float:
        returns = np.diff(self.simulated_paths) / self.simulated_paths[:, :-1]
        upside_returns = returns[returns >= 0]
        return np.std(upside_returns)

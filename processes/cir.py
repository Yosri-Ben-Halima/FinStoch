"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers
from utils.plotting import plot_simulated_paths

class CoxIngersollRoss:
    def __init__(self, S0: float, mu: float, sigma: float, theta: float, T: float, num_steps: int, num_paths: int) -> None:
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T / num_steps
        self._t = np.linspace(0, T, num_steps)

    def simulate(self) -> np.ndarray:
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            drift = self._theta * (self._mu - S[:, t - 1]) * self._dt
            diffusion = self._sigma * np.sqrt(S[:, t - 1]) * np.sqrt(self._dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion
            
            # Ensure non-negativity
            S[:, t] = np.maximum(S[:, t], 0)

        return S

    def plot(self, paths=None, ylabel='Value'):
        plot_simulated_paths(self._t, self.simulate, paths, title="Cox-Ingersoll-Ross Process", ylabel=ylabel)

    @property
    def S0(self) -> float:
        return self._S0

    @S0.setter
    def S0(self, value: float) -> None:
        self._S0 = value

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float) -> None:
        self._mu = value

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = value

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        self._T = value
        self._dt = value / self._num_steps
        self._t = np.linspace(0, value, self._num_steps)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_steps must be positive")
        self._num_steps = value
        self._dt = self._T / value
        self._t = np.linspace(0, self._T, value)

    @property
    def num_paths(self) -> int:
        return self._num_paths

    @num_paths.setter
    def num_paths(self, value: int) -> None:
        if value <= 0:
            raise ValueError("num_paths must be positive")
        self._num_paths = value

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def t(self) -> np.ndarray:
        return self._t

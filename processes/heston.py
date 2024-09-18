"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from typing import Tuple
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths

class HestonModel: 

    def __init__(self, S0: float, v0: float, mu: float, sigma: float, theta: float, kappa: float, rho: float, T: float, num_steps: float, num_paths: float) -> None:
        
        self._S0 = S0
        self._v0 = v0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._kappa = kappa
        self._rho = rho
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
        self._t = np.linspace(0, T, num_steps)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray] :
        """
        Simulates a path of the Heston model.

        ## Parameters:
        - self: Instance of HestonModel class.
    
        ## Returns:
        - A numpy array representing the asset price path.
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        v = np.zeros((self._num_paths, self._num_steps))
        v[:, 0] = self._v0

        for t in range(1, self._num_steps):
            Xs, Xv = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1), generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            L = np.array([[1, 0], [self._rho, np.sqrt(1 - self._rho**2)]])

            X = np.dot(L, np.array([Xs, Xv]))
            Ws = X[0]
            Wv = X[1]
            
            v[:, t] = np.maximum(v[:, t - 1] + self._kappa * (self._theta - v[:, t - 1]) * self._dt + self._sigma * np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Wv, 0)
            S[:, t] = S[:, t - 1] * np.exp((self._mu - 0.5 * v[:, t-1]) * self._dt + np.sqrt(v[:, t-1]) * np.sqrt(self._dt) * Ws)

        return S, v
    
    def plot(self, paths = None, ylabel = "Value", **kwargs):
        if kwargs.get('variance', False)==True:
            plot_simulated_paths(self._t, self.simulate, paths, title="Heston Model", ylabel="Variance", **kwargs)
        else:
            plot_simulated_paths(self._t, self.simulate, paths, title="Heston Model", ylabel=ylabel, **kwargs)
    
    @property
    def S0(self) -> float:
        return self._S0
    
    @S0.setter
    def S0(self, value: float) -> None:
        self._S0 = value
    
    @property
    def v0(self) -> float:
        return self._v0
    
    @v0.setter
    def v0(self, value: float) -> None:
        self._v0 = value

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
    def kappa(self) -> float:
        return self._kappa
    
    @kappa.setter
    def kappa(self, value: float) -> None:
        self._kappa = value
    
    @property
    def rho(self) -> float:
        return self._rho
    
    @rho.setter
    def rho(self, value: float) -> None:
        self._rho = value
    
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
        self._num_steps = value
        self._dt = self._T / value
        self._t = np.linspace(0, self._T, value)

    @property
    def num_paths(self) -> int:
        return self._num_paths
    
    @num_paths.setter
    def num_paths(self, value: int) -> None:
        self._num_paths = value

    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def t(self) -> np.ndarray:
        return self._t
    
    


    
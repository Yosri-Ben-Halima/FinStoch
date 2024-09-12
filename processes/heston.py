import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    def simulate(self) -> np.ndarray:
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
            
            v[:, t] = v[:, t - 1] + self._theta * (self._mu - v[:, t - 1]) * self._dt + self._sigma * np.sqrt(v[:, t - 1]) * np.sqrt(self._dt) * Wv
            S[:, t] = S[:, t - 1] * np.exp((self._mu - 0.5 * v[:, t]) * self._dt + self._sigma * np.sqrt(v[:, t]) * np.sqrt(self._dt) * Ws)

        return S, v
    


    
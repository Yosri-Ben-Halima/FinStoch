import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.random import generate_random_numbers # type: ignore
from utils.plotting import plot_simulated_paths

class MertonModel:

    def __init__(self, S0: float, mu: float, sigma: float, T: float, num_steps: float, num_paths: float, lambda_j: float, mu_j: float, sigma_j: float) -> None:
        
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
        self._lambda_j = lambda_j
        self._mu_j = mu_j
        self._sigma_j = sigma_j
        self._t = np.linspace(0, T, num_steps)

    def simulate(self) -> np.ndarray:
        """
        Simulates a path of the Merton jump diffusion model.

        ## Parameters:
        - self: Instance of MertonModel class.
        
        ## Returns:
        - A numpy array representing the asset price path.
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            N = generate_random_numbers('poisson', self._num_paths, lam=self._lambda_j * self._dt)
            J = np.zeros(self._num_paths)

            J[N > 0] = generate_random_numbers('normal', np.sum(N > 0), mean=self._mu_j, stddev=self._sigma_j)
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self._dt + self._sigma * np.sqrt(self._dt) * Z + J)

        return S

    def plot(self, paths = None, ylabel = 'Value'):
        plot_simulated_paths(self._t, self.simulate, paths, title="Merton Model", ylabel=ylabel)

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
    def lambda_j(self) -> float:
        return self._lambda_j

    @lambda_j.setter
    def lambda_j(self, value: float) -> None:
        self._lambda_j = value
    
    @property
    def mu_j(self) -> float:
        return self._mu_j
    
    @mu_j.setter
    def mu_j(self, value: float) -> None:
        self._mu_j = value
    
    @property
    def sigma_j(self) -> float:
        return self._sigma_j
    
    @sigma_j.setter
    def sigma_j(self, value: float) -> None:
        self._sigma_j = value
    
    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def t(self) -> np.ndarray:
        return self._t


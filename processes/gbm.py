import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths

class GeometricBrownianMotion:

    def __init__(self, S0: float, mu: float, sigma: float, T: float, num_steps: float, num_paths: float) -> None:
        
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
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
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self._dt + self._sigma * np.sqrt(self._dt) * Z )

        return S

    def plot(self, paths = None, ylabel = 'Price'):
        plot_simulated_paths(self.simulate, self._t, paths, title="Geometric Brownian Motion", ylabel=ylabel)

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
    def dt(self) -> float:
        return self._dt
    
    @property
    def t(self) -> np.ndarray:
        return self._t


np.random.seed(42)

# Parameters
S0 = 100
mu = 0.05
sigma = 0.2
T = 1.0 # 1 year
num_steps = 252 # 252 trading days in a year
num_paths = 10

gbm = GeometricBrownianMotion(S0, mu, sigma, T, num_steps, num_paths)
gbm.plot()



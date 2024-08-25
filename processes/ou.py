import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.random import generate_random_numbers # type: ignore
from utils.plotting import plot_simulated_paths

class OrnsteinUhlenbeck :

    def __init__(self, S0: float, mu: float, sigma: float, theta: float, T: float, num_steps: float, num_paths: float) -> None:
        
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
        self._t = np.linspace(0, T, num_steps)

    def simulate(self) -> float:
        
        S = np.zeros(( self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            drift = self._theta * (self._mu - S[:, t - 1]) * self._dt
            diffusion = self._sigma * np.sqrt(self._dt) * Z
            S[:, t] = S[:, t - 1] + drift + diffusion

        return S

    def plot(self, paths = None, ylabel = 'Price'):
        plot_simulated_paths(self.simulate, self._t, paths, title="Ornstein-Uhlenbeck Process", ylabel=ylabel)

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

    @sigma.setter
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
mu = 100
sigma = 0.5
theta = 0.5
T = 1.0 # 1 year
num_steps = 252 # 252 trading days in a year
num_paths = 10

gbm = OrnsteinUhlenbeck(S0, mu, sigma, theta, T, num_steps, num_paths)
gbm.plot()
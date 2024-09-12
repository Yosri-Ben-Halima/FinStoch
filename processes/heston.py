import numpy as np 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths

class HestonModel: 

    def __init__(self, S0: float, mu: float, sigma: float, theta: float, kappa: float, rho: float, T: float, num_steps: float, num_paths: float) -> None:
        
        self._S0 = S0
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
    
    

    
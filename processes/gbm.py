import numpy as np 
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range

class GeometricBrownianMotion:

    """
    A class to model the Geometric Brownian Motion (GBM) process for simulating asset prices.

    Attributes
    ----------
    S0 : float
        The initial value.
    mu : float
        The drift or mean return of the asset.
    sigma : float
        The volatility (standard deviation) of the asset.
    T : float
        The time horizon (in years) for the simulation.
    num_steps : int
        The number of time steps for the simulation.
    num_paths : int
        The number of paths to simulate.
    dt : float
        The time increment between steps (T / num_steps).
    start_date : str, optional
        The start date for the simulation. If not provided, time is treated numerically.
    t : np.ndarray
        The time or date range for the simulation steps.
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, T: float, num_steps: float, num_paths: float, start_date: str=None) -> None:
        """
        Initialize the parameters for the GBM model and set up the time or date steps.

        Parameters
        ----------
        S0 : float
            The initial value.
        mu : float
            The drift or mean return rate.
        sigma : float
            The volatility or standard deviation of returns.
        T : float
            The time horizon (in years).
        num_steps : int
            The number of time steps.
        num_paths : int
            The number of paths to simulate.
        start_date : str, optional
            Starting date for the simulation. If provided, time steps are based on dates. Otherwise, it's numerical.
        """
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        self._T = T
        self._num_steps = num_steps
        self._num_paths = num_paths
        self._dt = T/num_steps
        self._start_date = start_date
        if start_date is not None:
            self._t = generate_date_range(start_date, T, num_steps)
        else :
            self._t = np.linspace(0, T, num_steps)

    def simulate(self) -> np.ndarray:
        """
        Simulates a path of the GBM jump diffusion model.

        Parameters
        ----------
        self : GeometricBrownianMotion
            Instance of GeometricBrownianMotion class.
        
        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros((self._num_paths, self._num_steps))
        S[:, 0] = self._S0

        for t in range(1, self._num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self._dt + self._sigma * np.sqrt(self._dt) * Z )

        return S

    def plot(self, paths = None, ylabel = 'Value'):
        """
        Plots the simulated paths of the GBM model.

        Parameters
        ----------
        paths : optional
            If specified, the function will plot the given paths. Otherwise, it simulates new paths.
        ylabel : str, optional
            Label for the y-axis. Default is 'Value'.
        """
        plot_simulated_paths(self._t, self.simulate, paths, title="Geometric Brownian Motion", ylabel=ylabel)

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
        if self._start_date is not None:
            self._t = generate_date_range(self._start_date, value, self._num_steps)
        else :
            self._t = np.linspace(0, value, self._num_steps)

    @property
    def num_steps(self) -> int:
        return self._num_steps
    
    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._num_steps = value
        self._dt = self._T / value
        if self._start_date is not None:
            self._t = generate_date_range(self._start_date, self._T, value)
        else :
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
    
    @property
    def start_date(self) -> np.ndarray:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self._t = generate_date_range(value, self._T, self._num_steps)
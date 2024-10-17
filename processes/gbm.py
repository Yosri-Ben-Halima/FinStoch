"""
The `FinStoch.processes` module contains classes and methods for simulating various stochastic processes.
"""

import numpy as np 
from utils.random import generate_random_numbers 
from utils.plotting import plot_simulated_paths
from utils.timesteps import generate_date_range_with_granularity, date_range_duration


class GeometricBrownianMotion:

    """
    GeometricBrownianMotion
    =======================

    A class to model the Geometric Brownian Motion (GBM) process.

    Attributes
    ----------
    _S0 : float
        The initial value of the asset or variable being modeled (protected).
    _mu : float
        The annualized drift or mean return rate of the process (protected).
    _sigma : float
        The annualized volatility or standard deviation of the returns (protected).
    __T : float
        The total time horizon for the simulation, calculated from the date range (private).
    __num_steps : int
        The number of discrete time steps in the simulation, based on the date range (private).
    _num_paths : int
        The number of paths (scenarios) to simulate (protected).
    __dt : float
        The time increment between steps, calculated as T / num_steps (private).
    _start_date : str
        The start date for the simulation, provided as a string (protected).
    _end_date : str
        The end date for the simulation, provided as a string (protected).
    _granularity : str
        The time granularity for simulation steps, given as a Pandas frequency string (protected).
    __t : np.ndarray
        A NumPy array representing the discrete time steps or dates for the simulation (private).

    Methods
    -------
    simulate() -> np.ndarray
        Simulates multiple paths of the GBM process and returns the simulated paths as a 2D array.
    
    plot(paths=None, title="Geometric Brownian Motion", ylabel='Value', **kwargs) -> None
        Plots the simulated paths of the GBM process. If paths are provided, it will plot those paths; otherwise, it will simulate new paths and plot them.

    Properties
    ----------
    S0 :
        Getter and setter for the initial value of the variable.
    mu :
        Getter and setter for the drift rate.
    sigma :
        Getter and setter for the volatility.
    T :
        Getter for the time horizon of the simulation (private attribute).
    num_steps :
        Getter for the number of steps in the simulation (private attribute).
    num_paths :
        Getter and setter for the number of paths to simulate.
    dt :
        Getter for the time increment between steps (private attribute).
    t :
        Getter for the time steps or dates used in the simulation (private attribute).
    start_date :
        Getter and setter for the start date.
    end_date :
        Getter and setter for the end date.
    granularity :
        Getter and setter for the time granularity.
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, num_paths: float, start_date: str, end_date: str, granularity: str) -> None:
        """Initialize the Geometric Brownian Motion (GBM) process.

        Parameters
        ----------
        S0 : float
            The initial value of the asset.
        mu : float
            The annualized drift coefficient.
        sigma : float
            The annualized volatility coefficient.
        num_paths : int
            The number of paths to simulate.
        start_date : str
            The start date for the simulation (e.g., '2023-09-01').
        end_date : str
            The end date for the simulation (e.g., '2023-09-01 03:00:00').
        granularity : str
            The time granularity for each step in the simulation (e.g., '10T' for 10 minutes, 'H' for hours).
        """
        self._S0 = S0
        self._mu = mu
        self._sigma = sigma
        
        self._start_date = start_date
        self._end_date = end_date
        self._granularity = granularity
        self.__t = generate_date_range_with_granularity(self._start_date, self._end_date, self._granularity)
        
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
        
        self._num_paths = num_paths

    def simulate(self) -> np.ndarray:
        """
        Simulates paths of the GBM model.

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_paths, num_steps), where each row represents a simulated path of the variable.
        """
        S = np.zeros((self._num_paths, self.__num_steps))
        S[:, 0] = self._S0

        for t in range(1, self.__num_steps):
            Z = generate_random_numbers('normal', self._num_paths, mean=0, stddev=1)
            S[:, t] = S[:, t-1] * np.exp((self._mu - 0.5 * self._sigma**2) * self.__dt + self._sigma * np.sqrt(self.__dt) * Z )

        return S

    def plot(self, paths=None, title="Geometric Brownian Motion", ylabel='Value', fig_size: tuple=None, **kwargs):
        """
        Plot the simulated paths of the GBM model.

        Parameters
        ----------
        paths : np.ndarray, optional
            If provided, the method will plot the given paths. Otherwise, it simulates new paths.
        title : str, optional
            Title for the plot. Default is 'Geometric Brownian Motion'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Value'.
        fig_size : tuple, optional
            Size of the figure in inches. Default is None.
        **kwargs
            Additional keyword arguments for the plot, such as grid=True to show the grid.

        Returns
        -------
        None
        """
        plot_simulated_paths(self.__t, self.simulate, paths, title=title, ylabel=ylabel, fig_size=fig_size, grid=kwargs.get('grid', True))

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
        return self.__T

    @property
    def num_steps(self) -> int:
        return self.__num_steps
    
    @property
    def num_paths(self) -> int:
        return self._num_paths

    @num_paths.setter
    def num_paths(self, value: int) -> None:
        self._num_paths = value
    
    @property
    def dt(self) -> float:
        return self.__dt
    
    @property
    def t(self) -> np.ndarray:
        return self.__t

    @property
    def start_date(self) -> np.ndarray:
        return self._start_date
    
    @start_date.setter
    def start_date(self, value: str) -> None:
        self._start_date = value
        self.__t = generate_date_range_with_granularity(value, self._end_date, self._granularity)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps   
    
    @property
    def end_date(self) -> np.ndarray:
        return self._end_date
    
    @end_date.setter
    def end_date(self, value: str) -> None:
        self._end_date = value
        self.__t = generate_date_range_with_granularity(self._start_date, value, self._granularity)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
    
    @property
    def granularity(self) -> np.ndarray:
        return self._granularity
    
    @granularity.setter
    def granularity(self, value: str) -> None:
        self._granularity = value
        self.__t = generate_date_range_with_granularity(self._start_date, self._end_date, value)
        self.__T = date_range_duration(self.__t)
        self.__num_steps = len(self.__t)
        self.__dt = self.__T/self.__num_steps
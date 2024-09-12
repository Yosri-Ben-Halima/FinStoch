import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple   

def plot_simulated_paths(t, simulate_func=None, paths=None, title="Simulated Paths", ylabel = None, **kwargs):
    """
    Plots the simulated paths.

    ## Parameters:
    - t: Time array used for plotting.
    - simulate_func: The simulate method of the class instance.
    - paths: Optional, a numpy array representing the simulated paths.
    - title: Optional, the title for the plot.
    """
    if paths is None:
        paths = simulate_func()

    if isinstance(paths, Tuple) and all(isinstance(arr, np.ndarray) for arr in paths):
        S, v = paths
        if kwargs.get('variance', False)==True:
            paths = v
        else:
            paths = S

    plt.plot(t, paths.T)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

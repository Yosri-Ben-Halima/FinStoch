import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple   

def plot_simulated_paths(t, simulate_func=None, paths=None, title="Simulated Paths", ylabel=None, fig_size=None, **kwargs):
    """
    Plots the simulated paths.

    Parameters
    ----------
    t : array-like
        Time array used for plotting.
    simulate_func : callable, optional
        The simulate method of the class instance.
    paths : numpy.ndarray, optional
        A numpy array representing the simulated paths.
    title : str, optional
        The title for the plot.
    ylabel : str, optional
        The label for the y-axis.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None
    """
    if paths is None:
        paths = simulate_func()

    if isinstance(paths, Tuple) and all(isinstance(arr, np.ndarray) for arr in paths):
        S, v = paths
        if kwargs.get('variance', False) is True:
            paths = v
        else:
            paths = S

    if fig_size is not None:
        fig = plt.figure(figsize=fig_size)
    plt.plot(t, paths.T)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(kwargs.get('grid', True))
    plt.tight_layout()
    plt.show()

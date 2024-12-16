import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Callable, Union


def plot_simulated_paths(
    t,
    simulate_func: Optional[
        Callable[[], Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
    ] = None,
    paths: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
    title: str = "Simulated Paths",
    ylabel: Optional[str] = None,
    fig_size: Optional[Tuple] = None,
    **kwargs,
):
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
    assert (simulate_func is None) != (
        paths is None
    ), "Exactly one of 'simulate_func' or 'paths' must be provided."

    if (paths is None) and (simulate_func is not None):
        paths = simulate_func()

    if (
        (paths is not None)
        and isinstance(paths, tuple)
        and all(isinstance(arr, np.ndarray) for arr in paths)
    ):
        S, v = paths
        if kwargs.get("variance", False) is True:
            paths = v
        else:
            paths = S

    if fig_size is not None:
        plt.figure(figsize=fig_size)
    if paths is not None and isinstance(paths, np.ndarray):
        plt.plot(t, paths.T)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(kwargs.get("grid", True))
    plt.tight_layout()
    plt.show()

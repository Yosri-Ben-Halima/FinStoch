import matplotlib.pyplot as plt

def plot_simulated_paths(t, simulate_func=None, paths=None, title="Simulated Paths", ylabel = "Price"):
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

    plt.plot(t, paths.T)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

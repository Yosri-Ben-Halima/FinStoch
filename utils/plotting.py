import matplotlib.pyplot as plt

def plot_simulated_paths(simulate_func, t, paths=None, title="Simulated Paths", ylabel = "Price"):
    """
    Plots the simulated paths.

    ## Parameters:
    - simulate_func: The simulate method of the class instance.
    - t: Time array used for plotting.
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

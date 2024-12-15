import numpy as np


def generate_random_numbers(distribution: str, size: int, **kwargs) -> np.ndarray:
    """
    Generate an array of random numbers based on the specified distribution.

    Parameters
    ----------
    distribution : str
        The type of distribution to use ('normal' or 'poisson').
    size : int
        The number of random numbers to generate.
    **kwargs : Additional parameters for the distribution.

        For `normal` distribution:

            mean : float, optional
                The mean of the distribution (default is 0).
            stddev : float, optional
                The standard deviation of the distribution (default is 1).

        For `poisson` distribution:

            lam : float, optional
                The lambda (rate) parameter of the distribution (default is 1).

    Returns
    -------
    np.ndarray
        An array of random numbers generated from the specified distribution.

    Raises
    ------
    ValueError
        If the specified distribution type is unsupported.
    """

    if distribution == "normal":
        return np.random.normal(kwargs.get("mean", 0), kwargs.get("stddev", 1), size)
    elif distribution == "poisson":
        return np.random.poisson(kwargs.get("lam", 1), size)
    else:
        raise ValueError("Unsupported distribution type")

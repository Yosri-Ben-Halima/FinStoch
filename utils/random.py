import numpy as np

def generate_random_numbers(distribution: str, size: int, **kwargs) -> np.ndarray:
    if distribution == 'normal':
        return np.random.normal(kwargs.get('mean', 0), kwargs.get('stddev', 1), size)
    elif distribution == 'poisson':
        return np.random.poisson(kwargs.get('lam', 1), size)
    else:
        raise ValueError("Unsupported distribution type")

import numpy as np

from statistic import mean, std


def mean_squared_error(x, y):
    """Return the mean squared error
    Args:
        x (np.ndarray):
        y (np.ndarray):
    """
    if len(x) != len(y):
        raise ValueError("The size of array to be compared is different: {}, {}".format(len(x), len(y)))
    return mean((x - y) ** 2)


def mean_absolute_error(x, y):
    """Return the mean absolute error
    Args:
        x (np.ndarray):
        y (np.ndarray):
    """
    if len(x) != len(y):
        raise ValueError("The size of array to be compared is different: {}, {}".format(len(x), len(y)))
    return mean(np.abs(x - y))


def standardization(array):
    """Standardize the array.
    Args:
        array (np.ndarray):
    Return:
        np.ndarray: Standardized array
    """
    if len(array) == 0:
        return np.array([])
    e = mean(array)
    s = std(array)
    if s == 0:
        return None
    return (array - e) / s

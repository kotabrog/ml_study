import numpy as np
import math

from plot import plot_hist


def mean(array, multi=False):
    """Return the average value
    Args:
        array (np.ndarray): Target array
        multi (bool, optional):
            If True is specified, calculate per column
    """
    if len(array) == 0:
        return 0
    if multi:
        return np.sum(array, axis=0) / len(array)
    return np.sum(array) / len(array)


def var(array):
    """Return the variance value
    Args:
        array (np.ndarray): Target array
    """
    if len(array) == 0:
        return 0
    e = mean(array)
    return mean([(x - e) ** 2 for x in array])


def std(array):
    """Return the standard deviation value
    Args:
        array (np.ndarray): Target array
    """
    return np.sqrt(var(array))


def skew(array):
    """Return the skewness value
    Args:
        array (np.ndarray): Target array
    """
    if len(array) == 0:
        return 0
    s = std(array)
    if s == 0:
        return None
    e = mean(array)
    return mean([(x - e) ** 3 for x in array]) / s ** 3


def kurtosis(array):
    """Return the kurtosis value
    Args:
        array (np.ndarray): Target array
    """
    if len(array) == 0:
        return 0
    s = std(array)
    if s == 0:
        return None
    e = mean(array)
    return mean([(x - e) ** 4 for x in array]) / s ** 4 - 3


def moment(array, moment=1, mode="origin"):
    """Return the moment value
    Args:
        array (np.ndarray): Target array
        moment (int, optional): What order of moment
        mode: (string, optional):
            origin or mean.
            origin: Moment around the origin.
            mean: Moments around the expected value
    """
    if mode not in ["origin", "mean"]:
        raise ValueError("Specify origin or mean for the mode.")
    sub = 0 if mode == "origin" else mean(array)
    return mean([(x - sub) ** moment for x in array])


def median(array):
    """Return the median value
    Args:
        array (np.ndarray): Target array
    """
    array_len = len(array)
    sorted_array = np.sort(array)
    if array_len == 0:
        return 0
    elif array_len % 2:
        return sorted_array[array_len // 2]
    else:
        return (sorted_array[array_len // 2 - 1] + sorted_array[array_len // 2]) / 2


def partitive_point(array, alpha=0.5):
    """Return the alpha partitive point
    Args:
        array (np.ndarray): Target array
        alpha (float, optional): partitive point
    """
    if alpha < 0:
        alpha = 0
    elif alpha > 1:
        alpha = 1
    array_len = len(array)
    if array_len == 0:
        return 0
    sorted_array = np.sort(array)
    index = min(math.floor(array_len * alpha), array_len - 1)
    return sorted_array[index]


def mode(array):
    """Return the mode value
    Args:
        array (np.ndarray): Target array
    """
    if len(array) == 0:
        return None
    uniqs, counts = np.unique(array, return_counts=True)
    return uniqs[counts == np.amax(counts)]


def get_statistical_information(event_list, print_flag: bool = False):
    array = np.array(event_list)
    info = {}
    info['mean'] = mean(array)
    info['var'] = var(array)
    info['std'] = std(array)
    info['skew'] = skew(array)
    info['kurtosis'] = kurtosis(array)
    info['median'] = median(array)
    info['partitive_point1'] = partitive_point(array, alpha=0.25)
    info['partitive_point3'] = partitive_point(array, alpha=0.75)
    info['mode'] = mode(array)

    if print_flag:
        for key, value in info.items():
            print("{}: {}".format(key, value))
        plot_hist(array)

    return info


def cov(a, b):
    """Return the covariance value
    Args:
        a (np.ndarray):
        b (np.ndarray):
    """
    a, b = np.array(a), np.array(b)
    a_mean = mean(a)
    b_mean = mean(b)
    v = mean((a - a_mean) * (b - b_mean))
    return v


def corrcoef(a, b):
    """Return the correlation coefficient value
    Args:
        a (np.ndarray):
        b (np.ndarray):
    """
    a, b = np.array(a), np.array(b)
    cov_value = cov(a, b)
    a_std = std(a)
    b_std = std(b)
    if a_std == 0 or b_std == 0:
        return None
    return cov_value / a_std / b_std

def covariance_matrix(array):
    n = len(array[0])
    array = np.array(array)
    return np.array([[cov(array[:, i], array[:, j])
                    for i in range(n)]
                    for j in range(n)])

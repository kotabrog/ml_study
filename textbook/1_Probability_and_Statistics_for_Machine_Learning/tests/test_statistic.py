import numpy as np
from scipy import stats
import sys
import pytest

sys.path.append("lib")

from statistic import mean, var, std, skew,\
                          kurtosis, moment, median,\
                          partitive_point, mode,\
                          cov, corrcoef


def test_mean_corner_01():
    array = np.array([])
    assert mean(array) == 0

def test_mean_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert mean(array) == np.mean(array)

def test_mean_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert mean(array) == np.mean(array)

def test_mean_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert mean(array) == np.mean(array)


def test_var_corner_01():
    array = np.array([])
    assert var(array) == 0

def test_var_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert var(array) == np.var(array)

def test_var_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert var(array) == np.var(array)

def test_var_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert var(array) == np.var(array)


def test_std_corner_01():
    array = np.array([])
    assert std(array) == 0

def test_std_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert std(array) == np.std(array)

def test_std_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert std(array) == np.std(array)

def test_std_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert std(array) == np.std(array)


def test_skew_corner_01():
    array = np.array([])
    assert skew(array) == 0

def test_skew_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert np.abs(skew(array) - stats.skew(array)) < 1e-7

def test_skew_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert np.abs(skew(array) - stats.skew(array)) < 1e-7

def test_skew_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert np.abs(skew(array) - stats.skew(array)) < 1e-7


def test_kurtosis_corner_01():
    array = np.array([])
    assert kurtosis(array) == 0

def test_kurtosis_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert np.abs(kurtosis(array) - stats.kurtosis(array)) < 1e-7

def test_kurtosis_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert np.abs(kurtosis(array) - stats.kurtosis(array)) < 1e-7

def test_kurtosis_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert np.abs(kurtosis(array) - stats.kurtosis(array)) < 1e-7


def test_moment_corner_01():
    array = np.array([])
    assert moment(array) == 0

def test_moment_error_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        moment(array, mode="error")

def test_moment_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    mu_1 = moment(array, moment=1, mode="origin")
    mu_2 = moment(array, moment=2, mode="origin")
    mu_3 = moment(array, moment=3, mode="origin")
    mu_4 = moment(array, moment=4, mode="origin")
    assert np.abs(np.mean(array) - mu_1) < 1e-5
    assert np.abs(np.var(array) - (mu_2 - mu_1 ** 2)) < 1e-5
    assert np.abs(stats.skew(array) - (mu_3 - 3 * mu_2 * mu_1 + 2 * mu_1 ** 3) / np.power((mu_2 - mu_1 ** 2), 3 / 2)) < 1e-5
    assert np.abs(stats.kurtosis(array) - ((mu_4 - 4 * mu_3 * mu_1 + 6 * mu_2 * mu_1 ** 2 - 3 * mu_1 ** 4) / np.power((mu_2 - mu_1 ** 2), 2) - 3)) < 1e-5

def test_moment_normal_02():
    array = np.array(np.random.random_sample(10000) * 100)
    mu_1 = moment(array, moment=1, mode="origin")
    mu_2 = moment(array, moment=2, mode="origin")
    mu_3 = moment(array, moment=3, mode="origin")
    mu_4 = moment(array, moment=4, mode="origin")
    assert np.abs(np.mean(array) - mu_1) < 1e-5
    assert np.abs(np.var(array) - (mu_2 - mu_1 ** 2)) < 1e-5
    assert np.abs(stats.skew(array) - (mu_3 - 3 * mu_2 * mu_1 + 2 * mu_1 ** 3) / np.power((mu_2 - mu_1 ** 2), 3 / 2)) < 1e-5
    assert np.abs(stats.kurtosis(array) - ((mu_4 - 4 * mu_3 * mu_1 + 6 * mu_2 * mu_1 ** 2 - 3 * mu_1 ** 4) / np.power((mu_2 - mu_1 ** 2), 2) - 3)) < 1e-5

def test_moment_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    nu_2 = moment(array, moment=2, mode="mean")
    assert np.abs(np.var(array) - nu_2) < 1e-5


def test_median_corner_01():
    array = np.array([])
    assert median(array) == 0

def test_median_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert median(array) == np.median(array)

def test_median_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert median(array) == np.median(array)

def test_median_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert median(array) == np.median(array)


def test_partitive_point_corner_01():
    array = np.array([])
    assert partitive_point(array) == 0

def test_partitive_point_corner_02():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert partitive_point(array, -1) == 0

def test_partitive_point_corner_03():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert partitive_point(array, 2) == 5

def test_partitive_point_corner_04():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert partitive_point(array, 0) == 0

def test_partitive_point_corner_05():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert partitive_point(array, 1) == 5

def test_partitive_point_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert partitive_point(array, 0.5) == 3

def test_partitive_point_normal_02():
    array = np.array([0, 1, 6, 3, 4, 5])
    assert partitive_point(array, 0.4) == 3


def test_mode_corner_01():
    array = np.array([])
    assert mode(array) is None

def test_mode_normal_01():
    array = np.array([0, 0, 2, 1, 1])
    assert (mode(array) == np.array([0, 1])).all()

def test_mode_normal_02():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert(mode(array) == np.array([0, 1, 2, 3, 4, 5])).all()


def test_cov_corner_01():
    array = np.array([])
    assert cov(array, array) == 0

def test_cov_normal_01():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1, 0])
    assert np.abs(cov(a, b) - np.cov(a, b, bias=True)[0][1]) < 1e-5

def test_cov_normal_02():
    _mean = np.array([0, 0])
    _cov = np.array([[1, 0.5], [0.5, 1]])
    x, y = np.random.multivariate_normal(_mean, _cov, 1000).T
    assert np.abs(cov(x, y) - np.cov(x, y, bias=True)[0][1]) < 1e-5

def test_cov_normal_03():
    _mean = np.array([2, 1])
    _cov = np.array([[1, -0.95], [-0.95, 1]])
    x, y = np.random.multivariate_normal(_mean, _cov, 1000).T
    assert np.abs(cov(x, y) - np.cov(x, y, bias=True)[0][1]) < 1e-5


def test_corrcoef_corner_01():
    array = np.array([])
    assert corrcoef(array, array) is None

def test_corrcoef_corner_02():
    array = np.array([0, 0, 0])
    assert corrcoef(array, array) is None

def test_corrcoef_normal_01():
    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1, 0])
    assert np.abs(corrcoef(a, b) - np.corrcoef(a, b)[0][1]) < 1e-5

def test_corrcoef_normal_02():
    _mean = np.array([0, 0])
    _cov = np.array([[1, 0.5], [0.5, 1]])
    x, y = np.random.multivariate_normal(_mean, _cov, 1000).T
    assert np.abs(corrcoef(x, y) - np.corrcoef(x, y)[0][1]) < 1e-5

def test_corrcoef_normal_03():
    _mean = np.array([2, 1])
    _cov = np.array([[1, -0.95], [-0.95, 1]])
    x, y = np.random.multivariate_normal(_mean, _cov, 1000).T
    assert np.abs(corrcoef(x, y) - np.corrcoef(x, y)[0][1]) < 1e-5

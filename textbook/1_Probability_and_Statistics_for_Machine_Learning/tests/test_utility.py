import numpy as np
from sklearn import metrics
from scipy import stats
import sys
import pytest

sys.path.append("lib")

from utility import mean_squared_error, mean_absolute_error,\
                    standardization


def test_mean_squared_error_corner_01():
    array = np.array([])
    assert mean_squared_error(array, array) == 0

def test_mean_squared_error_error_01():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        mean_squared_error(x, y)

def test_mean_squared_error_normal_01():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6, 7])
    assert mean_squared_error(x, y) == metrics.mean_squared_error(x, y)

def test_mean_squared_error_normal_02():
    x = np.array(np.random.random_sample(100) * 100)
    y = np.array(np.random.random_sample(100) * 100)
    assert mean_squared_error(x, y) == metrics.mean_squared_error(x, y)

def test_mean_squared_error_normal_03():
    x = np.array(np.random.random_sample(10000) * 100)
    y = np.array(np.random.random_sample(10000) * 100)
    assert mean_squared_error(x, y) == metrics.mean_squared_error(x, y)


def test_mean_absolute_error_corner_01():
    array = np.array([])
    assert mean_absolute_error(array, array) == 0

def test_mean_absolute_error_error_01():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        mean_absolute_error(x, y)

def test_mean_absolute_error_normal_01():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6, 7])
    assert mean_absolute_error(x, y) == metrics.mean_absolute_error(x, y)

def test_mean_absolute_error_normal_02():
    x = np.array(np.random.random_sample(100) * 100)
    y = np.array(np.random.random_sample(100) * 100)
    assert mean_absolute_error(x, y) == metrics.mean_absolute_error(x, y)

def test_mean_absolute_error_normal_03():
    x = np.array(np.random.random_sample(10000) * 100)
    y = np.array(np.random.random_sample(10000) * 100)
    assert mean_absolute_error(x, y) == metrics.mean_absolute_error(x, y)


def test_standardization_corner_01():
    array = np.array([])
    assert len(standardization(array)) == 0

def test_standardization_corner_02():
    array = np.array([1, 1, 1])
    assert standardization(array) is None

def test_standardization_normal_01():
    array = np.array([0, 1, 2, 3, 4, 5])
    assert (standardization(array) == stats.zscore(array)).all()

def test_standardization_normal_02():
    array = np.array(np.random.random_sample(100) * 100)
    assert (standardization(array) == stats.zscore(array)).all()

def test_standardization_normal_03():
    array = np.array(np.random.random_sample(10000) * 100)
    assert (standardization(array) == stats.zscore(array)).all()

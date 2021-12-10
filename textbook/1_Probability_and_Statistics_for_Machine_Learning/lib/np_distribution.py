import numpy as np


def make_np_discrete_uniform(low, high):
    rng = np.random.default_rng()
    def func():
        return rng.integers(low, high)
    return func


def make_np_bernoulli(n, p):
    rng = np.random.default_rng()
    def func():
        return rng.binomial(n, p)
    return func


def make_np_hypergeometric(ngood, nbad, nsample):
    rng = np.random.default_rng()
    def func():
        return rng.hypergeometric(ngood, nbad, nsample)
    return func


def make_np_poisson(lam):
    rng = np.random.default_rng()
    def func():
        return rng.poisson(lam)
    return func


def make_np_negative_binomial(n, p):
    rng = np.random.default_rng()
    def func():
        return rng.negative_binomial(n, p)
    return func


def make_np_geometric(p):
    rng = np.random.default_rng()
    def func():
        return rng.geometric(p) - 1
    return func

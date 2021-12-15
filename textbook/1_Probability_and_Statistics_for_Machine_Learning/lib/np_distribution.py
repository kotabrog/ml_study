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


def make_np_continuous_uniform(low, high):
    rng = np.random.default_rng()
    def func():
        return rng.random() * (high - low) + low
    return func


def make_np_normal(loc, scale):
    rng = np.random.default_rng()
    def func():
        return rng.normal(loc, scale)
    return func


def make_np_gamma(shape, scale):
    rng = np.random.default_rng()
    def func():
        return rng.gamma(shape, scale)
    return func


def make_np_exponential(scale):
    rng = np.random.default_rng()
    def func():
        return rng.exponential(scale)
    return func


def make_np_chisquare(df):
    rng = np.random.default_rng()
    def func():
        return rng.chisquare(df)
    return func


def make_np_beta(a, b):
    rng = np.random.default_rng()
    def func():
        return rng.beta(a, b)
    return func


def make_np_standard_cauchy():
    rng = np.random.default_rng()
    def func():
        return rng.standard_cauchy()
    return func


def make_np_laplace(loc, scale):
    rng = np.random.default_rng()
    def func():
        return rng.laplace(loc, scale)
    return func


def make_np_standard_t(df):
    rng = np.random.default_rng()
    def func():
        return rng.standard_t(df)
    return func


def make_np_f(dfnum, dfden):
    rng = np.random.default_rng()
    def func():
        return rng.f(dfnum, dfden)
    return func
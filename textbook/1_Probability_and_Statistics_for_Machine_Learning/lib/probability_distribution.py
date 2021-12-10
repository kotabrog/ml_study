import random
import math
import numpy as np

def try_func_count(func, count: int):
    """Return a list of the results of executing func.
    Args:
        func (Callable):
        count (int):
    Return:
        list: a list of the results of executing func.
    """
    try_list = []
    for _ in range(count):
        try_list.append(func())
    return try_list


def dice():
    return random.randint(1, 6)


def discrete_uniform_mean(low, high):
    return (high + low - 1) / 2

def discrete_uniform_var(low, high):
    return ((high - low) ** 2 - 1) / 12


def make_bernoulli(n, p):
    rng = np.random.default_rng()
    def _bernoulli(n, p, x):
        return np.power(p, x) * np.power((1 - p), n - x) * math.comb(n, x)
    p = [_bernoulli(n, p, x) for x in range(0, n + 1)]
    x_list = list(range(0, n + 1))
    def func():
        return rng.choice(x_list, p=p)
    return func

def bernoulli_mean(n, p):
    return n * p

def bernoulli_var(n, p):
    return n * p * (1 - p)


def make_hypergeometric(ngood, nbad, nsample):
    rng = np.random.default_rng()
    def _hypergeometric(ngood, nbad, nsample, x):
        return math.comb(ngood, x) * math.comb(nbad, nsample - x) / math.comb(ngood + nbad, nsample)
    p = [_hypergeometric(ngood, nbad, nsample, x) for x in range(max(0, nsample - nbad), min(nsample, ngood) + 1)]
    x_list = list(range(max(0, nsample - nbad), min(nsample, ngood) + 1))
    def func():
        return rng.choice(x_list, p=p)
    return func

def hypergeometric_mean(ngood, nbad, nsample):
    return nsample * ngood / (ngood + nbad)

def hypergeometric_var(ngood, nbad, nsample):
    return nsample * ngood * nbad * (ngood + nbad - nsample) / ((ngood + nbad) ** 2 * (ngood + nbad - 1))


def make_poisson(lam):
    rng = np.random.default_rng()
    def _poisson(lam, x):
        return np.power(math.e, -lam) * np.power(lam, x) / math.factorial(x)
    p = []
    x = 0
    while 1 - sum(p) >= 1e-5:
        _p = _poisson(lam, x)
        if _p <= 0:
            break
        p.append(_p)
        x += 1
    _p = 1 - sum(p)
    if _p > 0:
        p.append(_p)
    x_list = list(range(0, len(p)))
    def func():
        return rng.choice(x_list, p=p)
    return func

def poisson_mean(lam):
    return lam

def poisson_var(lam):
    return lam


def make_negative_binomial(n, p):
    rng = np.random.default_rng()
    def _negative_binomial(n, p, x):
        return math.comb(n + x - 1, x) * np.power(p, n) * np.power(1 - p, x)
    pr = []
    x = 0
    while 1 - sum(pr) >= 1e-5:
        _pr = _negative_binomial(n, p, x)
        if _pr <= 0:
            break
        pr.append(_pr)
        x += 1
    _pr = 1 - sum(pr)
    if _pr > 0:
        pr.append(_pr)
    x_list = list(range(0, len(pr)))
    def func():
        return rng.choice(x_list, p=pr)
    return func

def negative_binomial_mean(n, p):
    return n * (1 - p) / p

def negative_binomial_var(n, p):
    return n * (1 - p) / p ** 2


def make_geometric(p):
    return make_negative_binomial(1, p)

def geometric_mean(p):
    return negative_binomial_mean(1, p)

def geometric_var(p):
    return negative_binomial_var(1, p)
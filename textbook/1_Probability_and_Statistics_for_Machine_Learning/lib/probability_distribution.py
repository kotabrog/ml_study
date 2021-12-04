import random


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

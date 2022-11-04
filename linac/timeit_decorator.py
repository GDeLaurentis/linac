import time
import functools


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def timeit(func):
    """Print the runtime of the decorated function as hours:minutes:seconds:milliseconds"""
    @functools.wraps(func)
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        s, ms = divmod(seconds * 1000, 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if "verbose" in kwargs.keys() and kwargs["verbose"] == True:
            print("\rTime elapsed in %s: %d:%02d:%02d:%03d.                                        " % (func.__name__, h, m, s, ms))
        return value
    return wrapper_timeit

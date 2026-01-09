import numpy
import random
import functools

from pycoretools import mapThreads


def test_multiprocessing_with_CUDA_context():
    from linac import cuda_row_reduce
    rows, cols = 500, 500
    A = numpy.array([[random.randint(0, (2 ** 31 - 1) - 1) for i in range(rows)] for j in range(cols)])
    cuda_row_reduce_2147483647 = functools.partial(cuda_row_reduce, field_characteristic=2 ** 31 - 1)
    mapThreads(cuda_row_reduce_2147483647, [A] * 10, mp_start_method='spawn', verbose=False)

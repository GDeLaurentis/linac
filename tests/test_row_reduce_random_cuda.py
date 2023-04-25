# -*- coding: utf-8 -*-

import numpy
import pytest

from linac.pycuda_row_reduce import cuda_row_reduce

try:
    import pycuda  # noqa
except ImportError:
    pycuda_found = False
else:
    pycuda_found = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize(
    "field_characteristic, matrix_size",
    [
        (0, 1024),
        (0, 2048),
        (0, 3072),
        (2 ** 31 - 1, 1024),
        (2 ** 31 - 1, 2048),
        (2 ** 31 - 1, 3072),
    ]
)
@pytest.mark.skipif(not pycuda_found, reason="pycuda not found")
def test_cuda_row_reduce(field_characteristic, matrix_size, verbose=True):
    shape = (matrix_size, matrix_size)
    if field_characteristic == 0:  # random complex matrix
        random_float_matrix1, random_float_matrix2 = numpy.random.rand(*shape), numpy.random.rand(*shape)
        random_matrix = random_float_matrix1 + 1j * random_float_matrix2
    else:
        random_matrix = numpy.random.randint(field_characteristic, size=(matrix_size, matrix_size))

    row_reduced_random_matrix = cuda_row_reduce(random_matrix, field_characteristic, verbose)
    if field_characteristic == 0:
        assert numpy.all(numpy.isclose(row_reduced_random_matrix - numpy.identity(matrix_size), numpy.zeros((matrix_size, matrix_size))))
    else:
        assert numpy.all(row_reduced_random_matrix == numpy.identity(matrix_size, dtype=int))

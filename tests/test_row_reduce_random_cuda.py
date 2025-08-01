# -*- coding: utf-8 -*-

import numpy
import pytest

from pathlib import Path

from linac.pycuda_row_reduce import cuda_row_reduce

try:
    import pycuda  # noqa
except ImportError:
    pycuda_found = False
else:
    pycuda_found = True

current_dir = Path(__file__).parent


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
        random_matrix = numpy.random.randint(field_characteristic, size=shape)

    row_reduced_random_matrix = cuda_row_reduce(random_matrix, field_characteristic, verbose)
    if field_characteristic == 0:
        assert numpy.all(numpy.isclose(row_reduced_random_matrix - numpy.identity(matrix_size), numpy.zeros(shape)))
    else:
        assert numpy.all(row_reduced_random_matrix == numpy.identity(matrix_size, dtype=int))


def test_random_cuda_row_reduce_known_res(verbose=True):
    matrix_size = 4097
    field_characteristic = 2 ** 31 - 19
    shape = (matrix_size, matrix_size)

    numpy.random.seed(0)
    random_matrix = numpy.random.randint(field_characteristic, size=shape, dtype=numpy.uint32)[:-1, :]

    rref = cuda_row_reduce(random_matrix, field_characteristic, verbose=False)

    known_res = numpy.load(current_dir / 'test_data' / 'known_result_4096_4097_2to32m19.npy')
    assert numpy.all(known_res == rref[:, -1])

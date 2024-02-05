# -*- coding: utf-8 -*-

import numpy
import pytest
import functools
import mpmath

from linac.row_reduce import row_reduce
from pyadic.finite_field import ModP


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_row_reduce_FF_ModP():
    matrix_size, field_characteristic = 100, 2 ** 31 - 1
    shape = (matrix_size, matrix_size)
    random_matrix = numpy.random.randint(field_characteristic, size=shape)
    random_matrix = numpy.vectorize(functools.partial(ModP, p=field_characteristic), otypes=[object])(random_matrix)

    row_reduced_random_matrix = row_reduce(random_matrix, pivoting=0, scaling=False, threshold=0, verbose=True)[0]

    assert numpy.all(row_reduced_random_matrix == numpy.identity(matrix_size, dtype=int))


def test_row_reduce_FF_int64():
    matrix_size, field_characteristic = 1000, 2 ** 31 - 1
    shape = (matrix_size, matrix_size)
    random_matrix = numpy.random.randint(field_characteristic, size=shape)

    row_reduced_random_matrix = row_reduce(random_matrix, pivoting=0, scaling=False, threshold=0, prime=field_characteristic, verbose=True)[0]

    assert numpy.all(row_reduced_random_matrix == numpy.identity(matrix_size, dtype=int))


@pytest.mark.parametrize("pivoting", [0, 1, 2, 3])
def test_row_reduce_complex128(pivoting):
    matrix_size = 1000
    shape = (matrix_size, matrix_size)
    random_matrix = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)

    row_reduced_random_matrix = row_reduce(random_matrix, pivoting=pivoting, scaling=True, verbose=True)[0]

    assert numpy.all(numpy.isclose(row_reduced_random_matrix - numpy.identity(matrix_size), numpy.zeros((matrix_size, matrix_size))))


def test_row_reduce_invalid_pivoting():
    matrix_size = 100
    shape = (matrix_size, matrix_size)
    random_matrix = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)

    with pytest.raises(ValueError):
        row_reduce(random_matrix, pivoting=4, scaling=True, verbose=True)[0]


def test_row_reduce_mpmath_mpc():
    mpmath.mp.dps = 64
    matrix_size = 100
    shape = (matrix_size, matrix_size)
    random_matrix = numpy.array(mpmath.randmatrix(*shape).tolist()) + 1j * numpy.array(mpmath.randmatrix(*shape).tolist())

    row_reduced_random_matrix = row_reduce(random_matrix, pivoting=1, scaling=True, threshold=10 ** -32, verbose=True)[0]

    row_reduced_random_matrix = row_reduced_random_matrix.astype('complex128')
    assert numpy.all(numpy.isclose(row_reduced_random_matrix - numpy.identity(matrix_size), numpy.zeros((matrix_size, matrix_size))))

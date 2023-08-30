# -*- coding: utf-8 -*-

import os
import numpy
import pytest

from linac import cuda_row_reduce, row_reduce
from linac.linear_algebra_tools import pivot_columns_from_row_reduced_echelon_form, drop_bottom_zero_rows, \
    canonical_kernel_from_row_reduced_echelon_form, row_reduced_echelon_form_from_canonical_kernel
from linac.sparse_matrix_tools import matrix_from_json_coo

try:
    import pycuda  # noqa
except ImportError:
    pycuda_found = False
else:
    pycuda_found = True


local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


known_pivots_small_test_matrix = [0, 1, 2, 3, 4, 5]
known_pivots_medium_test_matrix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 38, 39, 40, 41, 59]


@pytest.mark.parametrize(
    "cached_matrix_relative_path, field_characteristic, use_cuda, known_pivots, known_rref_shape, known_ck_shape",
    [
        ('/test_data/small_kernel_test_matrix_mpc.npy', 0, False,
         known_pivots_small_test_matrix, (6, 7), (7, 1)),
        ('/test_data/small_kernel_test_matrix_2147483647.npy', 2147483647, False,
         known_pivots_small_test_matrix, (6, 7), (7, 1)),
        pytest.param('/test_data/medium_kernel_test_matrix_mpc.npy', 0, True,
                     known_pivots_medium_test_matrix, (31, 86), (86, 55), marks=pytest.mark.skipif(not pycuda_found, reason="pycuda not found")),
        pytest.param('/test_data/medium_kernel_test_matrix_2147483647.npy', 2147483647, True,
                     known_pivots_medium_test_matrix, (31, 86), (86, 55), marks=pytest.mark.skipif(not pycuda_found, reason="pycuda not found")),
    ]
)
def test_pivots_and_kernels(cached_matrix_relative_path, field_characteristic, use_cuda, known_pivots, known_rref_shape, known_ck_shape):
    matrix = numpy.load(local_directory + cached_matrix_relative_path, allow_pickle=True)
    if use_cuda:
        row_reduced_matrix = cuda_row_reduce(matrix, field_characteristic=field_characteristic)
    else:
        row_reduced_matrix = row_reduce(matrix, pivoting=(0 if field_characteristic != 0 else 1))[0]
    row_reduced_echelon_form = drop_bottom_zero_rows(row_reduced_matrix)
    canonical_kernel = canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form)
    assert row_reduced_echelon_form.shape == known_rref_shape
    assert canonical_kernel.shape == known_ck_shape
    try:
        check = numpy.isclose(row_reduced_echelon_form.astype('complex'), row_reduced_echelon_form_from_canonical_kernel(
            canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form)).astype('complex')).all()
    except TypeError:
        check = numpy.isclose(row_reduced_echelon_form.astype('int64'), row_reduced_echelon_form_from_canonical_kernel(
            canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form)).astype('int64')).all()
    assert check
    assert row_reduced_echelon_form.shape[1] == canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form).shape[0]
    assert row_reduced_echelon_form.shape[0] + canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form).shape[1] == row_reduced_echelon_form.shape[1]
    assert pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form) == known_pivots


def test_pivots_and_kernels_with_large_fractions():
    rref = matrix_from_json_coo(local_directory + "/test_data/test_coo_large_fractions.json")
    assert numpy.all(rref == row_reduced_echelon_form_from_canonical_kernel(canonical_kernel_from_row_reduced_echelon_form(rref)))

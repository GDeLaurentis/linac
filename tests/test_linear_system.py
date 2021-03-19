# -*- coding: utf-8 -*-

import os
import sys
import numpy
import pytest
import re
import json
import functools

from io import StringIO
from fractions import Fraction
from lips.fields.finite_field import rationalise as rationalise_FF
from linac.linear_system_solver import iterative_gaussian_solver, rationalise
from linac.gmp_solver import mpc_matrix_to_gmp_matrix, single_iteration_gmp_solver, gmp_rationalise

try:
    import pycuda  # noqa
except ImportError:
    pycuda_found = False
else:
    pycuda_found = True

try:
    import gmpTools_found
except ImportError:
    gmpTools_found = False
else:
    gmpTools_found = True


local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Capturing(list):
    # from: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

known_pivots_small_test_matrix = [0, 1, 2, 3, 4, 5]
known_pivots_medium_test_matrix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 38, 39, 40, 41, 59]

with open(local_directory + "/test_data/known_linear_systems_solutions.json", 'r') as f:
    solutions = json.load(f)
solutions = tuple([tuple(map(lambda x: Fraction(*x), entry)) for entry in rat_sol] for rat_sol in solutions)


@pytest.mark.parametrize(
    "cached_matrix_relative_path, field_characteristic, use_cuda, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution",
    [
        # from 6pt split MHV tree, N/([16]⟨23⟩⟨34⟩[56]⟨2|1+6|5]s234) in [16] limit
        ('/test_data/small_linear_system_matrix_redundant.npy', 0, False, 6, 0, solutions[0]),
        # from 6pt split MHV tree, N/([16]⟨23⟩⟨34⟩[56]⟨2|1+6|5]s234) in s234 limit
        ('/test_data/small_linear_system_matrix_non_redundant.npy', 0, False, 0, 6, solutions[0]),
        # from 6pt split MHV tree, N/(⟨12⟩⟨16⟩[16]⟨23⟩⟨34⟩[34][45][56]s234s345) not in limit
        pytest.param('/test_data/medium_linear_system_matrix_non_redundant.npy', 0, True, 0, 819, solutions[1],
                     marks=pytest.mark.skipif(not pycuda_found, reason="pycuda not found")),
        # as previous one, but over a FF
        pytest.param('/test_data/medium_linear_system_matrix_non_redundant_131071.npy', 131071, True, 0, 819, solutions[1],
                     marks=pytest.mark.skipif(not pycuda_found, reason="pycuda not found")),
        # from ⟨12⟩[12]⟨34⟩[34]N/⟨1|3+4|2]⟨3|1+2|4]⟨5|1+2|6]Δ_135 (6g/pmpmpm 3mass triangle) - commented out, chached matrix is too big to commit to git
        # ('/test_data/large_linear_system_matrix_non_redundant.npy', 0, True, 0, 3011, solutions[2]),
    ]
)
def test_iterative_linear_solver(cached_matrix_relative_path, field_characteristic, use_cuda, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution):
    matrix = numpy.load(local_directory + cached_matrix_relative_path, allow_pickle=True)
    with Capturing() as output:
        if field_characteristic == 0:
            solution = iterative_gaussian_solver(matrix, use_gpu=use_cuda, pivoting=1, scaling=True, field_characteristic=0, verbose=True)
        else:
            solution = iterative_gaussian_solver(matrix, use_gpu=use_cuda, pivoting=0, scaling=False, field_characteristic=field_characteristic, verbose=True)
    print("\n".join(output))
    droped_redundants = list(map(int, re.findall(r"dropped_redundant: (\d+),", "".join(output))))
    assert droped_redundants[-1] == known_nbr_dropped_redundant
    dropped_zeros = list(map(int, re.findall(r"dropped_zero: (\d+),", "".join(output))))
    assert dropped_zeros[-1] == known_nbr_dropped_zero
    if field_characteristic == 0:
        rational_solution = list(map(rationalise, solution))
    else:
        rationalise_currentFF = functools.partial(rationalise_FF, n=field_characteristic)
        rational_solution = list(map(rationalise_currentFF, solution))
        rational_solution = [(entry, Fraction(0, 1)) for entry in rational_solution]
    assert rational_solution == known_rational_solution


@pytest.mark.parametrize(
    "cached_matrix_relative_path, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution",
    [
        ('/test_data/small_linear_system_matrix_non_redundant.npy', 0, 6, solutions[0]),  # 6pt split MHV tree, N/([16]⟨23⟩⟨34⟩[56]⟨2|1+6|5]s234) in s234 limit
    ]
)
@pytest.mark.skipif(not gmpTools_found, reason="gmpTools not found")
def test_gmp_linear_solver(cached_matrix_relative_path, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution):
    matrix = numpy.load(local_directory + cached_matrix_relative_path, allow_pickle=True)
    nInput = matrix.shape[0]
    gmp_matrix = mpc_matrix_to_gmp_matrix(matrix)
    with Capturing() as output:
        solution = single_iteration_gmp_solver(gmp_matrix, nInput)
    print("\n".join(output))
    droped_redundants = list(map(int, re.findall(r"dropped_redundant: (\d+),", "".join(output))))
    assert droped_redundants[-1] == known_nbr_dropped_redundant
    dropped_zeros = list(map(int, re.findall(r"dropped_zero: (\d+),", "".join(output))))
    assert dropped_zeros[-1] == known_nbr_dropped_zero
    rational_solution = list(map(gmp_rationalise, solution))
    assert rational_solution == known_rational_solution

# -*- coding: utf-8 -*-

import os
import sys
import numpy
import pytest
import re

from fractions import Fraction as Q
from io import StringIO
from linac.linear_system_solver import iterative_gaussian_solver, rationalise

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


@pytest.mark.parametrize(
    "cached_matrix_relative_path, field_characteristic, use_cuda, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution",
    [
        ('/test_data/small_linear_system_matrix_redundant.npy', 0, True, 6, 0,
         [(Q(1, 1), Q(0, 1)), (Q(3, 1), Q(0, 1)), (0, 0), (Q(3, 1), Q(0, 1)), (0, 0), (0, 0), (Q(1, 1), Q(0, 1)), (0, 0), (0, 0), (0, 0)])
    ]
)
def test_linear_solver(cached_matrix_relative_path, field_characteristic, use_cuda, known_nbr_dropped_redundant, known_nbr_dropped_zero, known_rational_solution):
    matrix = numpy.load(local_directory + cached_matrix_relative_path, allow_pickle=True)
    with Capturing() as output:
        solution = iterative_gaussian_solver(matrix)
    droped_redundants = list(map(int, re.findall(r"dropped_redundant: (\d+),", "".join(output))))
    assert droped_redundants[-1] == known_nbr_dropped_redundant
    dropped_zeros = list(map(int, re.findall(r"dropped_zero: (\d+),", "".join(output))))
    assert dropped_zeros[-1] == known_nbr_dropped_zero
    rational_solution = list(map(rationalise, solution))
    assert rational_solution == known_rational_solution

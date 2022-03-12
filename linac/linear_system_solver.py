#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe
# Created: 10/07/2018 - Updated: 23/02/2021

import bisect
import numpy

from fractions import Fraction
from linac.row_reduce import row_reduce
from linac.pycuda_row_reduce import cuda_row_reduce
from linac.linear_algebra_tools import non_pivot_columns_from_row_reduced_echelon_form, drop_bottom_zero_rows


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# See:
# https://en.wikipedia.org/wiki/LU_decomposition
# https://en.wikipedia.org/wiki/Rank_factorization

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def iterative_gaussian_solver(matrix, use_gpu=True, max_iterations=1, pivoting=1, scaling=True, field_characteristic=0, verbose=True):

    assert matrix.shape[0] + 1 == matrix.shape[1]
    nbr_unknowns = matrix.shape[0]
    dropped_redundant, dropped_zero = [], []
    last_dropped_redundant, last_dropped_zero = [], []

    for iteration_counter in range(0, max_iterations):

        dropped_old = sorted(dropped_redundant + dropped_zero)

        complementary_to_dropped_redundant = [entry for entry in range(matrix.shape[0]) if entry not in last_dropped_redundant]
        matrix = matrix[numpy.ix_(complementary_to_dropped_redundant, complementary_to_dropped_redundant + [matrix.shape[0]])]
        complementary_to_dropped_zero = [entry for entry in range(matrix.shape[0]) if entry not in last_dropped_zero]
        matrix = matrix[numpy.ix_(complementary_to_dropped_zero, complementary_to_dropped_zero + [matrix.shape[0]])]
        assert matrix.shape[0] + 1 == matrix.shape[1]

        if use_gpu is False:
            row_reduced_matrix, variable_order = row_reduce(matrix, pivoting=pivoting, scaling=scaling, reduced_echelon=True,
                                                            threshold=10 ** -9, prime=field_characteristic, verbose=verbose)
        else:
            row_reduced_matrix = cuda_row_reduce(matrix, field_characteristic=field_characteristic, verbose=verbose)
        row_reduced_matrix = drop_bottom_zero_rows(row_reduced_matrix)

        last_dropped_redundant = non_pivot_columns_from_row_reduced_echelon_form(row_reduced_matrix)
        if matrix.shape[0] in last_dropped_redundant:
            last_dropped_redundant.remove(matrix.shape[0])  # augmented column is not redundant

        reduced_solution = row_reduced_matrix[:, -1].tolist()
        last_dropped_zero = numpy.where(numpy.isclose(list(map(complex, reduced_solution)), 0))[0].tolist()
        reduced_solution = numpy.array(reduced_solution)[numpy.ix_([i for i in range(len(reduced_solution)) if i not in last_dropped_zero])].tolist()

        if pivoting == 2:  # fix ordering --- not working well yet
            last_dropped_redundant = sorted([variable_order[i] for i in last_dropped_redundant])
            last_dropped_zero = sorted([variable_order[i] for i in last_dropped_zero])
            reduced_solution, _ = list(map(list, zip(*sorted(zip(reduced_solution, variable_order), key=lambda s: s[1]))))

        # update indices of last dropped lists to match the original one
        tobekept = list(range(nbr_unknowns))
        for index in dropped_old[::-1]:
            tobekept.pop(index)
        for index in last_dropped_redundant[::-1]:
            bisect.insort(dropped_redundant, tobekept.pop(index))
        for index in last_dropped_zero[::-1]:
            bisect.insort(dropped_zero, tobekept.pop(index))

        dropped = sorted(dropped_redundant + dropped_zero)
        solution = [0 if i in dropped else reduced_solution.pop(0) for i in range(nbr_unknowns)]

        print("Iteration number {}: dropped_redundant: {}, dropped_zero: {}, dropped_total: {}.".format(
            iteration_counter + 1, len(dropped_redundant), len(dropped_zero), len(dropped)))

        if dropped == dropped_old or len(dropped) == nbr_unknowns:
            break

    return solution


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def rationalise(complex_number):
    """Approximate map from R to Q"""
    if complex_number == 0:
        real = 0
        imag = 0
    else:
        real = Fraction(str(complex_number.real)).limit_denominator(10 ** 3)
        imag = Fraction(str(complex_number.imag)).limit_denominator(10 ** 3)
    return (real, imag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def solve(row_reduced_matrix, rounding=True):
    """Performes backsubstitution in a LU decomposition."""
    # rationalisation can be performed at the same time - is this more stable? - deprecated for now
    unique_solution = []
    if len(row_reduced_matrix) < 1:
        return []
    for i in range(len(row_reduced_matrix[0]) - 1)[::-1]:
        unique_solution += [(row_reduced_matrix[i][len(row_reduced_matrix[0]) - 1] -
                            sum(entry1 * entry2 for entry1, entry2 in zip(unique_solution, row_reduced_matrix[i][::-1][1:]))) /
                            row_reduced_matrix[i][::-1][len(unique_solution) + 1]]
        if rounding is True and type(row_reduced_matrix[0][0]).__name__ not in ["ModP", "PAdic"]:
            unique_solution[-1] = (float(Fraction(str(unique_solution[-1].real)).limit_denominator(10 ** 3)) + 1j *
                                   float(Fraction(str(unique_solution[-1].imag)).limit_denominator(10 ** 3)))
    return unique_solution[::-1]

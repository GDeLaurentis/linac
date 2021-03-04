#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe
# Created: 10/07/2018 - Updated: 23/02/2021

import bisect
import numpy
import lips

from fractions import Fraction
from linac.timeit_decorator import timeit
from linac.row_reduce import row_reduce
from linac.pycuda_row_reduce import cuda_row_reduce
from linac.linear_algebra_tools import non_pivot_columns_from_row_reduced_echelon_form, drop_bottom_zero_rows


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# See:
# https://en.wikipedia.org/wiki/LU_decomposition
# https://en.wikipedia.org/wiki/Rank_factorization

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def iterative_gaussian_solver(matrix, use_gpu=True, max_iterations=10, pivoting=1, scaling=True, field_characteristic=0, verbose=True):

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
        last_dropped_zero = numpy.where(numpy.isclose(reduced_solution, 0))[0].tolist()
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
        real = Fraction(complex_number.real).limit_denominator(10 ** 3)
        imag = Fraction(complex_number.imag).limit_denominator(10 ** 3)
    return (real, imag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def Solve(RowReducedMatrix, rounding=True):
    """Performes backsubstitution in a LU decomposition."""
    # Rationalisation can be performed at the same time - is this more stable?
    UniqueSolution = []
    if len(RowReducedMatrix) < 1:
        return []
    for i in range(len(RowReducedMatrix[0]) - 1)[::-1]:
        UniqueSolution += [(RowReducedMatrix[i][len(RowReducedMatrix[0]) - 1] -
                            sum(entry1 * entry2 for entry1, entry2 in zip(UniqueSolution, RowReducedMatrix[i][::-1][1:]))) /
                           RowReducedMatrix[i][::-1][len(UniqueSolution) + 1]]
        if rounding is True and type(RowReducedMatrix[0][0]) not in [lips.fields.ModP, lips.fields.PAdic]:
            UniqueSolution[-1] = (float(Fraction(str(UniqueSolution[-1].real)).limit_denominator(10 ** 3)) + 1j *
                                  float(Fraction(str(UniqueSolution[-1].imag)).limit_denominator(10 ** 3)))
    return UniqueSolution[::-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def InversionUsingSingleGMPGaussianElimination(GMPMatrix, nInput):
    dropped_redundant = GMPRowReduce(GMPMatrix)
    UniqueSolution = GMPSolve(GMPMatrix, nInput, dropped_redundant)
    Solution = OriginalSolution(nInput, dropped_redundant, UniqueSolution)

    dropped_zero = [i for i, entry in enumerate(Solution) if (i not in dropped_redundant and
                                                              (entry.real.makeFraction() == 0 or abs(entry.real.makeFraction()) < 10 ** -9) and
                                                              (entry.imag.makeFraction() == 0 or abs(entry.imag.makeFraction()) < 10 ** -9))]

    print("Nbr dropped redundant: {}, Nbr dropped zero: {}, Nbr dropped total: {}.".format(
        len(dropped_redundant), len(dropped_zero), len(dropped_redundant) + len(dropped_zero)))
    return Solution


@timeit
def GMPRowReduce(GMPMatrix):
    """GMP row reduction from SpinorSolve"""
    dropped = GMPMatrix.rowReduceInPlace()
    return dropped


def GMPSolve(GMPMatrix, nInput, dropped):  # This is rowreduced, but not strictly upper triangular
    """Back substitution."""
    UniqueSolution = []
    for i in range(nInput - len(dropped))[::-1]:
        UniqueRow = []
        for j in range(nInput + 1):
            if j not in dropped:
                UniqueRow += [GMPMatrix[i, j]]
        UniqueSolution += [(GMPMatrix[i, nInput] -
                            sum(entry1 * entry2 for entry1, entry2 in zip(UniqueSolution, UniqueRow[::-1][1:]))) /
                           UniqueRow[::-1][len(UniqueSolution) + 1]]
    return UniqueSolution[::-1]


def OriginalSolution(len_original_matrix, dropped, UniqueSolution):
    # reconstruct the solution in terms of the original (non-minimal) ansatz
    Solution = []
    for i in range(len_original_matrix):
        if i in dropped:
            Solution.append(0)
        else:
            Solution.append(UniqueSolution.pop(0))
    return Solution


def GMPRationalise(complex_number):
    if complex_number == 0:
        real, imag = 0, 0
    else:
        real = complex_number.real.makeFraction()
        imag = complex_number.imag.makeFraction()
        if abs(real) < 10 ** -9:
            real = 0
        if abs(imag) < 10 ** -9:
            imag = 0
    return (real, imag)

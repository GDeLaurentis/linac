#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

import mpmath
import gmpTools

from linac.timeit_decorator import timeit


def single_iteration_gmp_solver(gmp_matrix, nInput):
    dropped_redundant = gmp_row_reduce(gmp_matrix)
    unique_solution = gmp_solve(gmp_matrix, nInput, dropped_redundant)
    solution = original_solution(nInput, dropped_redundant, unique_solution)

    dropped_zero = [i for i, entry in enumerate(solution) if (i not in dropped_redundant and
                                                              (entry.real.makeFraction() == 0 or abs(entry.real.makeFraction()) < 10 ** -9) and
                                                              (entry.imag.makeFraction() == 0 or abs(entry.imag.makeFraction()) < 10 ** -9))]

    print("Single iteration: dropped_redundant: {}, dropped_zero: {}, dropped_total: {}.".format(
        len(dropped_redundant), len(dropped_zero), len(dropped_redundant) + len(dropped_zero)))
    return solution


@timeit
def gmp_row_reduce(gmp_matrix):
    """GMP row reduction from SpinorSolve"""
    dropped = gmp_matrix.rowReduceInPlace()
    return dropped


def gmp_solve(gmp_matrix, nInput, dropped):  # This is rowreduced, but not strictly upper triangular
    """Back substitution."""
    unique_solution = []
    for i in range(nInput - len(dropped))[::-1]:
        unique_row = []
        for j in range(nInput + 1):
            if j not in dropped:
                unique_row += [gmp_matrix[i, j]]
        unique_solution += [(gmp_matrix[i, nInput] -
                            sum(entry1 * entry2 for entry1, entry2 in zip(unique_solution, unique_row[::-1][1:]))) /
                            unique_row[::-1][len(unique_solution) + 1]]
    return unique_solution[::-1]


def original_solution(len_original_matrix, dropped, unique_solution):
    # reconstruct the solution in terms of the original (non-minimal) ansatz
    solution = []
    for i in range(len_original_matrix):
        if i in dropped:
            solution.append(0)
        else:
            solution.append(unique_solution.pop(0))
    return solution


def gmp_rationalise(complex_number):
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


def cgmp_to_mpc(cgmp_nbr):
    return mpmath.mpc(str(cgmp_nbr.real), str(cgmp_nbr.imag))


def mpc_to_cgmp(mpc_nbr):
    return gmpTools.CGMP(str(mpc_nbr.real), str(mpc_nbr.imag))


def mpc_matrix_to_gmp_matrix(matrix):
    nRows, nColumns = matrix.shape
    gmp_matrix = gmpTools.GMPCmatrix(nRows, nColumns)  # build the gmptools matrix, this uses C++ code
    for i in range(nRows):
        for j in range(nColumns):
            gmp_matrix[i, j] = mpc_to_cgmp(matrix[i][j])
        # else:                 # throw away the row to avoid doubling memory usage
        #     matrix[i] = []    # (Note: might not be necessary if copy on write works as expected)
    return gmp_matrix

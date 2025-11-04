#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe
# Created: 10/07/2018 - Updated: 12/02/2021

import numpy
import fractions

from pyadic.finite_field import extended_euclidean_algorithm

from .timeit_decorator import timeit


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@timeit
def row_reduce(matrix, pivoting=1, scaling=True, reduced_echelon=True, threshold=10 ** -9, prime=None, verbose=False):
    """
    Row reduction to (reduced) echelon form on the cpu using numpy.
    The matrix is kept of whatever object type it is made of, unless prime is specified.
    In the latter case the computation happens over numpy.int64 in a finite field of cardinality prime.
    """
    if prime == 0:
        prime = None

    if pivoting not in [0, 1, 2, 3]:
        raise ValueError("Invalid pivoting.")
    pivoting_type = ("no" if pivoting == 0 else "partial" if pivoting == 1 else "rook" if pivoting == 2 else "total")

    (i, j) = (0, 0)
    variable_order = list(range(matrix.shape[1]))

    if scaling is True:
        row_scales = abs(matrix).max(axis=1, keepdims=True)
        matrix = matrix / row_scales
    if prime is not None:
        if matrix.dtype is numpy.dtype(object):  # if dtype is object make sure all fractions are converted to the correct finite field value
            matrix = numpy.vectorize(lambda x: (x.numerator * extended_euclidean_algorithm(x.denominator, prime)[0]) % prime if isinstance(x, fractions.Fraction) else x)(matrix)
        matrix = matrix.astype('int64')

    while i < matrix.shape[0] and j < matrix.shape[1]:

        if verbose:
            print("\rrow_reduce with {} pivoting at line {}/{}".format(pivoting_type, i, matrix.shape[0] - 1), end="")

        if pivoting == 1:  # partial pivoting
            pivot_row = numpy.argmax(numpy.abs(matrix[i:, j])) + i
            matrix[[i, pivot_row], :] = matrix[[pivot_row, i], :]  # permute row

        elif pivoting > 1:

            if pivoting == 2:  # rook pivoting
                pivot_row, pivot_column = numpy.unravel_index(numpy.argmax(numpy.abs(matrix[i:, j:]) > 0.1), matrix[i:, j:].shape) + numpy.array([i, j])
            elif pivoting == 3:  # total pivoting
                pivot_row, pivot_column = numpy.unravel_index(numpy.argmax(numpy.abs(matrix[i:, j:])), matrix[i:, j:].shape) + numpy.array([i, j])

            matrix[[i, pivot_row], :] = matrix[[pivot_row, i], :]  # permute row
            matrix[:, [j, pivot_column]] = matrix[:, [pivot_column, j]]  # permute column
            variable_order[j], variable_order[pivot_column] = variable_order[pivot_column], variable_order[j]  # keep track of variable order

        if abs(matrix[i][j]) > threshold:
            if prime is not None:
                s, t, gcd = extended_euclidean_algorithm(matrix[i][j], prime)
                if gcd != 1:
                    raise ZeroDivisionError("Inverse of {} mod {} does not exist. Are you sure {} is prime?".format(matrix[i][j], prime, prime))
                matrix[i, :] = matrix[i, :] * s % prime
            else:
                matrix[i, :] = matrix[i, :] / matrix[i][j]
            matrix[i + 1:, j:] = matrix[i + 1:, j:] - matrix[i + 1:, j: j + 1] * matrix[i: i + 1, j:]
            if reduced_echelon is True:
                matrix[:i, j:] = matrix[:i, j:] - matrix[:i, j: j + 1] * matrix[i: i + 1, j:]
            (i, j) = (i + 1, j + 1)
        else:
            j = j + 1

        if prime is not None:
            matrix[:, j:] = matrix[:, j:] % prime

    return matrix, variable_order

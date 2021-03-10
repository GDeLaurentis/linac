#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe
# Created: 12/02/2021

import numpy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form):
    try:
        return [numpy.where(numpy.isclose(row_reduced_echelon_form[i, :].astype('complex'), 1) == True)[0][0] for i in range(row_reduced_echelon_form.shape[0])]  # noqa
    except TypeError:
        return [numpy.where(numpy.isclose(row_reduced_echelon_form[i, :].astype('int64'), 1) == True)[0][0] for i in range(row_reduced_echelon_form.shape[0])]  # noqa


def non_pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form):
    pivot_columns = pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form)
    return [index for index in range(row_reduced_echelon_form.shape[1]) if index not in pivot_columns]


def non_pivot_columns_from_canonical_kernel(canonical_kernel):
    try:
        return [numpy.where(numpy.isclose(canonical_kernel[:, i].astype('complex'), -1) == True)[0][-1] for i in range(canonical_kernel.shape[1])]  # noqa
    except TypeError:
        return [numpy.where(numpy.isclose(canonical_kernel[:, i].astype('int64'), -1) == True)[0][-1] for i in range(canonical_kernel.shape[1])]  # noqa


def pivot_columns_from_canonical_kernel(canonical_kernel):
    non_pivot_columns = non_pivot_columns_from_canonical_kernel(canonical_kernel)
    return [index for index in range(canonical_kernel.shape[0]) if index not in non_pivot_columns]


def isclose_or_greater(x, y, *args, **kwargs):
    return (numpy.isclose(x, y, *args, **kwargs) or x > y)


def drop_bottom_zero_rows(row_reduced_matrix):
    while not isclose_or_greater(float(abs(row_reduced_matrix[-1]).max()), 1):
        row_reduced_matrix = row_reduced_matrix[:-1]
    return row_reduced_matrix


def canonical_kernel_from_row_reduced_echelon_form(row_reduced_echelon_form):
    """Returns the kernel or null-space of a matrix in canonical form. The input should be in reduced echelon form."""
    identity = numpy.identity(row_reduced_echelon_form.shape[1] - row_reduced_echelon_form.shape[0], dtype=int)
    non_pivot_columns = non_pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form)
    pivot_columns = pivot_columns_from_row_reduced_echelon_form(row_reduced_echelon_form)
    incomplete_row_echelon_form = numpy.delete(row_reduced_echelon_form, pivot_columns, axis=1)  # drop left interweaved identity
    scrambled_kernel = numpy.concatenate((incomplete_row_echelon_form, -identity), axis=0)  # append negative identity to the bottom
    permutation = [(pivot_columns + non_pivot_columns).index(index) for index in range(len(pivot_columns + non_pivot_columns))]
    canonical_kernel = scrambled_kernel[permutation, :]  # interweave bottom negative identity
    return canonical_kernel


def row_reduced_echelon_form_from_canonical_kernel(canonical_kernel):
    """Returns the row reduced echelon form the kernel kernel."""
    identity = numpy.identity(canonical_kernel.shape[0] - canonical_kernel.shape[1], dtype=int)
    non_pivot_columns = non_pivot_columns_from_canonical_kernel(canonical_kernel)
    pivot_columns = pivot_columns_from_canonical_kernel(canonical_kernel)
    incomplete_kernel = numpy.delete(canonical_kernel, non_pivot_columns, axis=0)  # drop bottom interweaved negative identity
    scrambled_row_reduced_echelon_form = numpy.concatenate((identity, incomplete_kernel), axis=1)  # prepend identity on left
    permutation = [(pivot_columns + non_pivot_columns).index(index) for index in range(len(pivot_columns + non_pivot_columns))]
    rref = scrambled_row_reduced_echelon_form[:, permutation]  # interweave left identity
    return rref

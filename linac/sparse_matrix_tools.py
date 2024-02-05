#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe
# Created: 03/09/2021

import json
import numpy
import operator

from fractions import Fraction

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def sparsity(matrix):
    """String representing the % of zeros in matrix."""
    matrix = numpy.atleast_2d(matrix)
    return "%.3f" % ((1 - numpy.count_nonzero(matrix) / operator.mul(*matrix.shape)) * 100) + "%"


def coo_from_matrix(matrix):
    """COO (Coordinate list) dictionary: {(row, column): value}."""
    rows, columns = matrix.shape
    coo = {}
    for row_index in range(rows):
        for column_index in range(columns):
            value = matrix[row_index, column_index]
            if value == 0:
                continue
            coo[(row_index, column_index)] = value
    if (coo == {} or rows != max([row for row, column in coo.keys()]) + 1 or
       columns != max([column for row, column in coo.keys()]) + 1):
        coo[(rows - 1, columns - 1)] = matrix[rows - 1, columns - 1]
    return coo


def matrix_from_coo(coo):
    """Converts a COO dictionary back to a numpy array."""
    rows = max([row for row, column in coo.keys()]) + 1
    columns = max([column for row, column in coo.keys()]) + 1
    matrix = numpy.zeros((rows, columns), dtype=object)
    for key in coo.keys():
        matrix[key] = coo[key]
    return matrix


def coo_from_json_coo(file_name, dtype=Fraction):
    """Loads coo from json containing coo dictionary."""
    with open(file_name, 'r') as f:
        data = json.load(f)
    coo = {}
    for key in data.keys():
        coo[eval(key)] = dtype(data[key])
    return coo


def matrix_from_json_coo(file_name, dtype=Fraction):
    """Loads matrix from json containing coo dictionary."""
    coo = coo_from_json_coo(file_name, dtype=dtype)
    matrix = matrix_from_coo(coo)
    return matrix


def json_coo_from_coo(coo, file_name):
    coo = {str(key): str(val) for key, val in coo.items()}
    with open(file_name, 'w') as f:
        json.dump(coo, f)


def json_coo_from_matrix(matrix, file_name):
    coo = coo_from_matrix(matrix)
    json_coo_from_coo(coo, file_name)


def plain_txt_coo_from_coo(coo, file_name):
    with open(file_name, "w+") as f:
        for key, val in coo.items():
            f.write(f"{key[0]} {key[1]} {val}\n")


def plain_txt_coo_from_matrix(matrix, file_name):
    coo = coo_from_matrix(matrix)
    plain_txt_coo_from_coo(coo, file_name)


def coo_from_plain_txt_coo(file_name, dtype=Fraction):
    """Loads coo from json containing coo dictionary."""
    with open(file_name, 'r') as f:
        lines = f.readlines()
    coo = {}
    for line in lines:
        key0, key1, val = line.split(" ")
        coo[(eval(key0), eval(key1))] = dtype(val)
    return coo


def matrix_from_plain_txt_coo(file_name, dtype=Fraction):
    """Loads coo from json containing coo dictionary."""
    coo = coo_from_plain_txt_coo(file_name, dtype=dtype)
    matrix = matrix_from_coo(coo)
    return matrix

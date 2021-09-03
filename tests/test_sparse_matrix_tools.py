# -*- coding: utf-8 -*-

import os
import numpy

from linac.sparse_matrix_tools import sparsity, coo_from_matrix, matrix_from_coo, matrix_from_json_coo, json_coo_from_matrix


local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_coo():
    matrix = matrix_from_json_coo(local_directory + "/test_data/test_coo.json")
    assert sparsity(matrix) == '95.180%'
    assert numpy.all(matrix_from_coo(coo_from_matrix(matrix)) == matrix)


def test_json_coo():
    matrix = matrix_from_json_coo(local_directory + "/test_data/test_coo.json")
    json_coo_from_matrix(matrix, "/tmp/test_coo.json")
    matrix_reloaded = matrix_from_json_coo("/tmp/test_coo.json")
    assert numpy.all(matrix == matrix_reloaded)

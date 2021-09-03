# -*- coding: utf-8 -*-

import numpy
import random

from linac.linear_algebra_tools import permutation_matrix_from_permutation, invert_permutation


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_random_permutation():
    import random
size = random.randint(50, 100)
random_permutation = tuple(random.sample(range(size), size))
assert numpy.all(permutation_matrix_from_permutation(random_permutation) @
                 permutation_matrix_from_permutation(invert_permutation(random_permutation)) ==
                 numpy.identity(len(random_permutation)))

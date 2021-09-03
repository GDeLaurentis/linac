# -*- coding: utf-8 -*-

import numpy
import random

from linac.linear_algebra_tools import permutation_to_permutation_matrix, invert_permutation


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_random_permutation():
    import random
size = random.randint(50, 100)
random_permutation = tuple(random.sample(range(size), size))
assert numpy.all(permutation_to_permutation_matrix(random_permutation) @
                 permutation_to_permutation_matrix(invert_permutation(random_permutation)) ==
                 numpy.identity(len(random_permutation)))

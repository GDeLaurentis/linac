import numpy
import pytest

from linac import GALOIS_FOUND
from fractions import Fraction as Q
from linac import ColumnVectorSpace


@pytest.mark.parametrize("use_galois", [False, True])
def test_column_vector_space(use_galois):
    if use_galois and not GALOIS_FOUND:
        pytest.skip("galois not installed")

    def CVS(mat):
        return ColumnVectorSpace(numpy.array(mat), use_galois=use_galois)

    a = CVS([[1, 1],
             [1, -1]])
    b = CVS([[1, 0],
             [0, 1]])
    c = CVS([[1],
             [0]])

    assert a in b and b in a
    assert (a & c) in c and c in (a & c)


def test_column_vector_space_rat():
    a = ColumnVectorSpace(numpy.array([
        [Q(1), Q(1)],
        [Q(1), Q(-1)]
    ], dtype=object), prime=None)
    b = ColumnVectorSpace(numpy.array([
        [Q(1), Q(0)],
        [Q(0), Q(1)]
    ], dtype=object), prime=None)
    c = ColumnVectorSpace(numpy.array([
        [Q(1), ],
        [Q(0), ]
    ], dtype=object), prime=None)
    assert a in b and b in a
    assert a & c in c and c in a & c

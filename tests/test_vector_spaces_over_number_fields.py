import numpy

from linac import ColumnVectorSpace


def test_column_vector_space():
    a = ColumnVectorSpace(numpy.array([
        [1, 1],
        [1, -1]
    ]))
    b = ColumnVectorSpace(numpy.array([
        [1, 0],
        [0, 1]
    ]))
    c = ColumnVectorSpace(numpy.array([
        [1, ],
        [0, ]
    ]))
    assert a in b and b in a
    assert a & c in c and c in a & c

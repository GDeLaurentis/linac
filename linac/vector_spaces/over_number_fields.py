# Author: Giuseppe

import fractions
import numpy

from ..row_reduce import row_reduce
from ..linear_algebra_tools import drop_bottom_zero_rows, pivot_columns_from_row_reduced_echelon_form, canonical_kernel_from_row_reduced_echelon_form


class ColumnVectorSpace(object):
    """Column vector space over number fields (FF and Q supported atm)."""

    def __init__(self, matrix, prime=2147483647):
        if prime is not None:
            assert prime <= 2 ** 31 - 1
            self.matrix = matrix % prime
            self.matrix = self.matrix.astype(numpy.int32)
            self.prime = prime
        else:
            self.matrix = numpy.vectorize(fractions.Fraction, otypes='O')(matrix)
            self.prime = None

    @property
    def dim(self):
        return self.matrix.shape[1]

    def __str__(self):
        return f"Vector space of dim. {self.dim}"

    def __repr__(self):
        return str(self)

    def __and__(self, other):
        A, B = self.matrix, other.matrix
        AmB = numpy.block([A, -B]) if self.prime is None else numpy.block([A, -B]) % self.prime
        # AmBrref = cuda_row_reduce(AmB, field_characteristic=self.prime)
        AmBrref, _ = row_reduce(AmB, pivoting=1, scaling=False, threshold=0, prime=self.prime)
        AmBrref = drop_bottom_zero_rows(AmBrref)
        AmBkernel = canonical_kernel_from_row_reduced_echelon_form(AmBrref)
        AmBkernel = AmBkernel if self.prime is None else AmBkernel % self.prime
        x = AmBkernel[:A.shape[1]].astype(object)  # use object to avoid overflow
        y = AmBkernel[A.shape[1]:].astype(object)
        A = A.astype(object)
        B = B.astype(object)
        if self.prime is None:
            assert numpy.all(A @ x == B @ y)
        else:
            assert numpy.all((A @ x) % self.prime == (B @ y) % self.prime)
        w = (A @ x) if self.prime is None else (A @ x) % self.prime
        return ColumnVectorSpace(w, self.prime)

    def __contains__(self, other):
        A, B = self.matrix, other.matrix
        AB = numpy.block([A, B])
        # rref = cuda_row_reduce(AB, field_characteristic=self.prime)
        rref, _ = row_reduce(AB, pivoting=1, scaling=False, threshold=0, prime=self.prime)
        rref = drop_bottom_zero_rows(rref)
        pivots = pivot_columns_from_row_reduced_echelon_form(rref)
        return pivots == list(range(A.shape[1]))

    def __eq__(self, other):
        # return self in other and other in self   # this doesn't assume canonical kernel form
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(self.matrix.flatten()))  # this assumes canonical kernel form

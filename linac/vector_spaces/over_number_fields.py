# Author: Giuseppe

import fractions
import numpy
import warnings

from pyadic.finite_field import vec_ModP

from .._optional import CUDA_FOUND, GALOIS_FOUND
from ..pycuda_row_reduce import cuda_row_reduce
from ..row_reduce import row_reduce
from ..linear_algebra_tools import drop_bottom_zero_rows, pivot_columns_from_row_reduced_echelon_form, canonical_kernel_from_row_reduced_echelon_form


class ColumnVectorSpace(object):
    """Column vector space over number fields (finite fields and rationals currently supported)."""

    def __init__(self, matrix, prime=2147483647, use_galois=None, use_gpu=None, verbose=False):
        """
        Parameters
        ----------
        matrix : 2D array-like
            Matrix whose columns span the vector space.
        prime : int or None, default 2147483647
            If an integer, arithmetic is performed modulo prime. If None, entries are converted to fractions.Fraction and arithmetic is performed over Q.
        use_galois : bool or None, default None
            If True, use the galois library for finite-field arithmetic on the CPU. If None, use galois when available. This option is ignored when GPU acceleration is used.
        use_gpu : bool or None, default None
            If True, use GPU acceleration for row reduction. If False, force CPU row reduction.
            If None, use GPU acceleration when CUDA/PyCUDA is available. GPU acceleration is not supported for operations in Q.
        verbose : bool, default False
            If True, print verbose output.
        """

        use_galois_was_explicit = use_galois is not None
        use_gpu_was_explicit = use_gpu is not None

        if use_galois is None:
            use_galois = GALOIS_FOUND  # default behaviour

        if use_gpu is None:
            use_gpu = CUDA_FOUND  # default behaviour

        if prime is None and use_gpu:
            if use_gpu_was_explicit:
                warnings.warn(
                    "GPU acceleration is not supported for rational coefficients; "
                    "falling back to CPU row reduction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            use_gpu = False

        if use_gpu and not CUDA_FOUND:
            raise RuntimeError("use_gpu=True requested but CUDA/PyCUDA is not available.")

        if use_gpu and use_galois:
            if use_galois_was_explicit:
                warnings.warn(
                    "use_galois=True is ignored when use_gpu=True; "
                    "using plain NumPy integer arrays for GPU row reduction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            use_galois = False

        self.use_galois = use_galois
        self.uses_gpu = use_gpu
        self.verbose = verbose

        if prime is not None:
            self.prime = prime
            self.matrix = vec_ModP(prime)(matrix)

            if use_galois:
                if not GALOIS_FOUND:
                    raise RuntimeError("use_galois=True requested but galois is not installed.")
                self.matrix = self.matrix.astype(int).view(self.GFp)
            else:
                assert prime <= 2 ** 31 - 1
                self.matrix = self.matrix.astype(numpy.int32)
        else:
            self.matrix = numpy.vectorize(fractions.Fraction, otypes='O')(matrix)
            self.prime = None

    @property
    def GFp(self):
        if not self.use_galois:
            raise ValueError("Tried to access GFp but use_galois is False.")
        import galois
        return galois.GF(self.prime)

    @property
    def dim(self):
        return self.matrix.shape[1]

    def __str__(self):
        return f"Vector space of dim. {self.dim}"

    def __repr__(self):
        return str(self)

    def __and__(self, other):
        A, B = self.matrix, other.matrix
        AmB = numpy.block([A, -B]) if (self.prime is None or self.use_galois) else numpy.block([A, -B]) % self.prime
        if self.uses_gpu:
            AmBrref = cuda_row_reduce(AmB, field_characteristic=self.prime)
        else:
            AmBrref, _ = row_reduce(AmB, pivoting=1, scaling=False, threshold=0, prime=self.prime)
        AmBrref = drop_bottom_zero_rows(AmBrref)
        AmBkernel = canonical_kernel_from_row_reduced_echelon_form(AmBrref)
        AmBkernel = AmBkernel if (self.prime is None or self.use_galois) else AmBkernel % self.prime
        if self.prime is None or not self.use_galois:
            x = AmBkernel[:A.shape[1]].astype(object)  # use object to avoid overflow
            y = AmBkernel[A.shape[1]:].astype(object)
            A = A.astype(object)
            B = B.astype(object)
        else:
            x = vec_ModP(self.prime)(AmBkernel[:A.shape[1]]).astype(int).view(self.GFp)
            y = vec_ModP(self.prime)(AmBkernel[A.shape[1]:]).astype(int).view(self.GFp)
        if self.prime is None or self.use_galois:
            assert numpy.all(A @ x == B @ y)
        else:
            assert numpy.all((A @ x) % self.prime == (B @ y) % self.prime)
        w = (A @ x) if (self.prime is None or self.use_galois) else (A @ x) % self.prime
        return ColumnVectorSpace(w, self.prime, use_galois=self.use_galois, use_gpu=self.uses_gpu, verbose=self.verbose, )

    def __contains__(self, other):
        A, B = self.matrix, other.matrix
        AB = numpy.block([A, B])
        if self.uses_gpu:
            rref = cuda_row_reduce(AB, field_characteristic=self.prime)
        else:
            rref, _ = row_reduce(AB, pivoting=1, scaling=False, threshold=0, prime=self.prime)
        rref = drop_bottom_zero_rows(rref)
        pivots = pivot_columns_from_row_reduced_echelon_form(rref)
        return pivots == list(range(A.shape[1]))

    def __eq__(self, other):
        # return self in other and other in self   # this doesn't assume canonical kernel form
        return hash(self) == hash(other)           # might want to revisit this, it is not the intuitive equality

    def __hash__(self):
        return hash(tuple(self.matrix.flatten().tolist()))  # this assumes canonical kernel form

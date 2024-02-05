# Author: Giuseppe

import functools
import numpy

from ..pycuda_row_reduce import cuda_row_reduce
from ..row_reduce import row_reduce
from ..linear_algebra_tools import drop_bottom_zero_rows, pivot_columns_from_row_reduced_echelon_form
from ..tensor_function import tensor_function
from .tools import mapThreads   # !! WARNING: This requires a better solution !!


class VectorSpaceOfFunctions(object):
    """Vector Spaces of Fraction Fields of Polynomial (Quotient) Rings"""  # actually, rationality of the function is not required

    def __init__(self, functions_evaluator, input_generator, verbose=True, use_gpu=True):
        """functions_evaluator should return a 1d numpy array"""
        self.uses_gpu = use_gpu
        self.all_functions_evaluator = functions_evaluator
        self.input_generator = input_generator
        self.__get_pivots__()
        self.basis_functions = self.all_functions_evaluator[self.pivots]
        if verbose:
            print(f"Instantiated vector space of dimension {self.dim}.")

    @property
    def dim(self):
        return len(self.pivots)

    def __add__(self, other):
        def combined_functions(*args, **kwargs):
            return numpy.block([self.basis_functions(*args, **kwargs), other.basis_functions(*args, **kwargs)])
        return VectorSpaceOfFunctions(tensor_function(combined_functions), self.input_generator, verbose=self.verbose, use_gpu=self.uses_gpu)

    def __repr__(self):
        return f"Vector space of rational functions of dimension {self.dim}."

    def __get_pivots__(self, iteration_step=20, max_iteration=50):
        iteration = 0
        random_points = [self.input_generator(i) for i in range(iteration * iteration_step, iteration * iteration_step + iteration_step)]
        A = self._numerical_matrix_repr(self.all_functions_evaluator, tuple(random_points))
        if self.uses_gpu:
            rref = cuda_row_reduce(A, field_characteristic=2 ** 31 - 1)
        else:
            rref, _ = row_reduce(A, scaling=False, threshold=0, prime=2 ** 31 - 1, )
        while not numpy.all(rref[-1, :] == 0):
            iteration += 1
            if iteration > max_iteration:
                raise Exception(f"Vector space exceeded dimension {iteration_step * max_iteration}")
            random_points = [self.input_generator(i) for i in range(iteration * iteration_step, iteration * iteration_step + iteration_step)]
            _A = self._numerical_matrix_repr(self.all_functions_evaluator, tuple(random_points))
            A = numpy.block([[A], [_A]])
            if self.uses_gpu:
                rref = cuda_row_reduce(A, field_characteristic=2 ** 31 - 1)
            else:
                rref, _ = row_reduce(A, scaling=False, threshold=0, prime=2 ** 31 - 1, )
        rref = drop_bottom_zero_rows(rref)
        self.pivots = pivot_columns_from_row_reduced_echelon_form(rref)

    def __contains__(self, other):
        """Is other in self? other should be a function or a vector space of rational functions."""
        if isinstance(other, VectorSpaceOfFunctions) and other.dim > self.dim:
            return False
        random_points = [self.input_generator(i) for i in range(self.dim + 1)]
        A0 = self._numerical_matrix_repr(self.basis_functions, tuple(random_points))
        if isinstance(other, VectorSpaceOfFunctions):
            A1 = self._numerical_matrix_repr(other.basis_functions, tuple(random_points))
        else:
            def other_as_tensor(*args, **kwargs):
                return numpy.array([other(*args, **kwargs)])
            A1 = self._numerical_matrix_repr(other_as_tensor, tuple(random_points))
        A = numpy.block([A0, A1])
        if self.uses_gpu:
            rref = cuda_row_reduce(A, field_characteristic=2 ** 31 - 1)
        else:
            rref, _ = row_reduce(A, scaling=False, threshold=0, prime=2 ** 31 - 1, )
        rref = drop_bottom_zero_rows(rref)
        pivots = pivot_columns_from_row_reduced_echelon_form(rref)
        assert all([index in pivots for index in range(self.dim)])
        return len(pivots) == self.dim

    @staticmethod
    @functools.lru_cache()
    def _numerical_matrix_repr(functions, random_points):
        matrix = mapThreads(functions, random_points, Cores=5, UseParallelisation=True, verbose=False)
        matrix = numpy.array(matrix).astype('uint32')
        return matrix

    def close_under_symmetries(self, symmetries):
        """Symmetries are applied to first positional arguments, which should define a image function under the symmetry."""
        def self_closed_under_symmetries(*args, **kwargs):
            result_closed_under_symmetries = self.basis_functions(*args, **kwargs).tolist()
            for symmetry in symmetries:
                result_closed_under_symmetries += self.basis_functions(args[0].image(symmetry), *args[1:], **kwargs).tolist()
            return numpy.array(result_closed_under_symmetries)
        return VectorSpaceOfFunctions(tensor_function(self_closed_under_symmetries), self.input_generator, verbose=self.verbose, use_gpu=self.uses_gpu)

#   __  __      _       _     _                 _
#  |  \/  |__ _| |_ _ _(_)_ _| |   ___  __ _ __| |___ _ _
#  | |\/| / _` |  _| '_| \ \ / |__/ _ \/ _` / _` / -_) '_|
#  |_|  |_\__,_|\__|_| |_/_\_\____\___/\__,_\__,_\___|_|

# Author: Giuseppe
# Created: 10/07/2018
# Updated: 09/04/2022

import os
import numpy

from operator import mul
from functools import reduce
from copy import deepcopy
from pycoretools import flatten

from .timeit_decorator import timeit

local_directory = os.path.dirname(os.path.abspath(__file__))


@timeit
def cuda_load_matrix(bases, lindices, shape, field_characteristic=0, _real=None, _mod64=None):
    """Load matrix on gpu by parellelizing multiplication - requires uniform degree entries"""
    nRows, nColumns = shape

    import pycuda.driver as cuda
    import pycuda.autoinit                    # noqa
    from pycuda.compiler import SourceModule  # noqa
    from linac.pycuda_tools import cuda_set_vars_and_get_funcs, folded_number_of_columns

    if field_characteristic > 0:
        _real = True  # doesn't matter
        if _mod64 or (_mod64 is None and field_characteristic > 2 ** 31 - 1):
            dtype = 'uint64'
            _mod64 = True
        else:
            dtype = 'uint32'
            _mod64 = False
    else:
        _mod64 = False  # doesn't matter
        if _real or (_real is None and numpy.max(numpy.abs(numpy.vectorize(lambda x: x.imag)(bases[0]))) == 0):
            dtype = 'float64'
            _real = True
        else:  # runs with complex128
            dtype = 'complex128'
            _real = False

    # Compile Cuda Code
    exec(str(cuda_set_vars_and_get_funcs(
        path_to_cuda_script=local_directory + "/matrix_loader.cu",
        FIELD_CHARACTERISTIC=field_characteristic,
        REAL=int(_real), MOD64=int(_mod64), EXTENSION='ULL' if _mod64 else 'u',
        BASIS_LENGTH=len(bases[0]), NBR_ROWS=nRows,
        NBR_COLUMNS=nColumns, DEGREE=len(lindices[0]), )),
        locals(), globals())

    # Push Bases And Indices To Device
    bases = bases.astype(dtype)
    bases_gpu = cuda.mem_alloc(bases.size * bases.dtype.itemsize)
    cuda.memcpy_htod(bases_gpu, bases)
    lindices_gpu = cuda.mem_alloc(lindices.size * lindices.dtype.itemsize)
    cuda.memcpy_htod(lindices_gpu, lindices)

    # Push Empty Matrix To Device
    matrix_cpu = numpy.zeros((nRows, nColumns), dtype=dtype)
    matrix_gpu = cuda.mem_alloc(matrix_cpu.size * matrix_cpu.dtype.itemsize)
    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    # Load Matrix On GPGPU
    CudaLoadMatrix(matrix_gpu, bases_gpu, lindices_gpu, block=(folded_number_of_columns(nColumns), 1, 1), grid=(nRows, 1))  # noqa

    # Pull Loaded Matrix Back To Host
    cuda.memcpy_dtoh(matrix_cpu, matrix_gpu)

    return matrix_cpu


def load_matrices(prefactors, ansatze, points, use_cuda=True):
    if isinstance(points, dict):
        field = points['field']
    else:
        field = points[0].field
    if field.characteristic == 0:
        if field.name in ['mpf', 'R']:
            python_type = float
        else:
            python_type = complex
    else:
        python_type = int

    # pad all ansatz monomials to equal length
    ansatze = deepcopy(ansatze)
    for i, ansatz in enumerate(ansatze):
        max_length = max([len(sublist) for sublist in ansatz], default=0)
        ansatz = numpy.array([sublist + ['1'] * (max_length - len(sublist)) for sublist in ansatz], dtype=str)
        ansatz = ansatz.tolist()
        ansatze[i] = ansatz

    # build numerical bases
    variables = ['1'] + sorted(list(set(flatten(ansatze))), key=lambda x: len(x))
    if isinstance(points, dict):
        bases_monomials = numpy.vectorize(python_type, otypes='O')(numpy.array([points[var] for var in variables]).T.copy())
    else:
        bases_monomials = numpy.array([[python_type(point(var)) for var in variables] for point in points])

    # complete build matrices
    As = []
    for i, prefactor in enumerate(prefactors):
        # complete basis with prefactor
        if isinstance(points, dict):
            bases_prefactor = numpy.vectorize(python_type, otypes='O')(numpy.array([points[prefactor]]).T.copy())
        else:
            bases_prefactor = numpy.array([[python_type(point(prefactor)) if isinstance(prefactor, str)
                                            else 1 if prefactor(point) == 1
                                            else python_type(prefactor(point))] for point in points])
        bases = numpy.block([bases_monomials, bases_prefactor])
        result_is_vector = bases.ndim < 2
        bases = numpy.atleast_2d(bases)

        if field.name == "padic":
            bases = bases % field.characteristic

        lindices = numpy.zeros((len(ansatze[i]), (len(ansatze[i][0]) if len(ansatze[i]) > 0 else 0) + 1), dtype='uint32')
        for j, product_of_variables in enumerate(ansatze[i]):
            k = -1
            for k, variable in enumerate(product_of_variables):
                lindices[j, k] = variables.index(variable)
            lindices[j, k + 1] = len(bases[0]) - 1
        if len(lindices) > 0:
            if use_cuda:
                As += [cuda_load_matrix(bases, lindices, (bases.shape[0], lindices.shape[0]),
                                        field_characteristic=field.characteristic, _real=(python_type is float))]
            else:
                if field.characteristic > 0:
                    As += [numpy.array([[reduce(lambda a, b: int(a) * int(b) % field.characteristic,  # int avoids overflow
                                                [base[index] for index in indices]) for indices in lindices] for base in bases],
                                       dtype='uint64' if field.characteristic > 2 ** 31 - 1 else 'uint32')]
                else:
                    As += [numpy.array([[reduce(mul, [base[index] for index in indices]) for indices in lindices] for base in bases])]
        else:
            As += [numpy.empty((bases.shape[0], lindices.shape[0]), dtype='uint32')]
    if result_is_vector:
        return [A[0] for A in As]
    return As

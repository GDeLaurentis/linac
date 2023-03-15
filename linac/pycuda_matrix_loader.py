#   __  __      _       _     _                 _
#  |  \/  |__ _| |_ _ _(_)_ _| |   ___  __ _ __| |___ _ _
#  | |\/| / _` |  _| '_| \ \ / |__/ _ \/ _` / _` / -_) '_|
#  |_|  |_\__,_|\__|_| |_/_\_\____\___/\__,_\__,_\___|_|

# Author: Giuseppe
# Created: 10/07/2018
# Updated: 09/04/2022

import os
import numpy
# import operator
# import copy
# import functools
from linac.timeit_decorator import timeit

local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# if settings.UseGpu is True and settings.field.name in ["mpc", "finite field"]:
#     print("\rLoading square matrix on gpu.                             ", end="\n")
#     rectangular_matrix = cuda_load_square_matrix(bases, lindices, (nRows, oAnsatz.nInput))
#     rectangular_matrix = rectangular_matrix.tolist()
# else:
#     print("\rLoading square matrix on cpu.                             ", end="\n")
#     rectangular_matrix = mapThreads(evaluate_row, lindices, bases, UseParallelisation=settings.UseParallelisation, Cores=settings.Cores)
# print("\rFinished loading the square matrix.                  ", end="")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@timeit
def cuda_load_matrix(bases, lindices, shape, field_characteristic=0):
    """Load matrix on gpu by parellelizing multiplication - requires uniform degree entries"""
    nRows, nColumns = shape

    import pycuda.driver as cuda
    import pycuda.autoinit                    # noqa
    from pycuda.compiler import SourceModule  # noqa
    from linac.pycuda_tools import cuda_set_vars_and_get_funcs, folded_number_of_columns

    # Compile Cuda Code
    exec(str(cuda_set_vars_and_get_funcs(path_to_cuda_script=local_directory + "/matrix_loader.cu",
                                         FIELD_CHARACTERISTIC=field_characteristic, BASIS_LENGTH=len(bases[0]), NBR_ROWS=nRows,
                                         NBR_COLUMNS=nColumns, DEGREE=len(lindices[0]))), locals(), globals())

    # Push Bases And Indices To Device
    bases = bases.astype('complex128' if field_characteristic == 0 else 'uint32')
    bases_gpu = cuda.mem_alloc(bases.size * bases.dtype.itemsize)
    cuda.memcpy_htod(bases_gpu, bases)
    lindices_gpu = cuda.mem_alloc(lindices.size * lindices.dtype.itemsize)
    cuda.memcpy_htod(lindices_gpu, lindices)

    # Push Empty Matrix To Device
    matrix_cpu = numpy.zeros((nRows, nColumns), dtype=('complex128' if field_characteristic == 0 else 'uint32'))
    matrix_gpu = cuda.mem_alloc(matrix_cpu.size * matrix_cpu.dtype.itemsize)
    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    # Load Matrix On GPGPU
    CudaLoadMatrix(matrix_gpu, bases_gpu, lindices_gpu, block=(folded_number_of_columns(nColumns), 1, 1), grid=(nRows, 1))  # noqa

    # Pull Loaded Matrix Back To Host
    cuda.memcpy_dtoh(matrix_cpu, matrix_gpu)

    return matrix_cpu


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# def evaluate_row(lindices, basis):
#     return [functools.reduce(operator.mul, [basis[index] for index in indices]) for indices in lindices]

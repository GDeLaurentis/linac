# Author: Giuseppe
# Created: 11/02/2021

import os
import math
import time
import numpy

from linac.pycuda_tools import cuda_set_vars_and_get_funcs, number_of_foldings, folded_number_of_columns, round_to_multiple_of
from linac.timeit_decorator import timeit

local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@timeit
def cuda_row_reduce(matrix, field_characteristic=0, verbose=False, _real=None, _mod64=None):
    r"""
    Args:
        matrix (2D numpy.ndarray):
            The matrix to be cast to row-reduced echelon form.
        field_characteristic (int, optional, default: 0):
            The characteristic p of the field, or 0 for real and complex matrices.
        verbose (bool, optional, default: False):
            If True, prints additional information.
    """

    import pycuda.driver as cuda
    import pycuda.autoinit                     # noqa
    from pycuda.compiler import SourceModule   # noqa

    if True:  # debug - check cuda context per thread/process, useful when parallelising
        print(f"[pid {os.getpid()}] ctx={cuda.Context.get_current()}")

    # Compile Cuda Code - sets Cuda* Functions
    NbrRows, NbrColumns = matrix.shape

    # Push Matrix To Device
    if field_characteristic > 0:
        _real = True  # doesn't matter
        if _mod64 or (_mod64 is None and field_characteristic > 2 ** 32 - 1):
            matrix_cpu = matrix.astype('uint64')
            _mod64 = True
        else:
            matrix_cpu = matrix.astype('uint32')
            _mod64 = False
    else:
        _mod64 = False  # doesn't matter
        if _real or (_real is None and numpy.max(numpy.vectorize(lambda x: x.imag)(matrix)) == 0):
            matrix_cpu = numpy.vectorize(lambda x: x.real)(matrix).astype("float64")
            _real = True
        else:  # runs with complex128
            matrix_cpu = matrix.astype("complex128")
            _real = False
    width = NbrColumns * matrix_cpu.dtype.itemsize
    height = NbrRows

    # Need alignment in order to access uint32 as 4 vector
    access_size = 16
    if width % access_size == 0:
        pad = 0
    else:
        pad = access_size - width % access_size
    pitch = width + pad
    matrix_gpu = cuda.mem_alloc(pitch * height)

    EffNbrColumns = pitch // matrix_cpu.dtype.itemsize

    # Copy matrix to device
    matrix_copier = cuda.Memcpy2D()
    matrix_copier.set_src_host(matrix_cpu)
    matrix_copier.set_dst_device(matrix_gpu)
    matrix_copier.width_in_bytes = width
    matrix_copier.src_pitch = width
    matrix_copier.dst_pitch = pitch
    matrix_copier.height = height
    matrix_copier(aligned=True)

    exec(str(cuda_set_vars_and_get_funcs(path_to_cuda_script=local_directory + "/row_reduce.cu",
                                         NBR_ROWS=NbrRows, NBR_COLUMNS=EffNbrColumns, TRUE_NBR_COLUMNS=NbrColumns,
                                         FIELD_CHARACTERISTIC=field_characteristic, REAL=int(_real), MOD64=int(_mod64),
                                         EXTENSION='ULL' if _mod64 else 'u')),
         locals(), globals())

    if field_characteristic == 0:  # Set The Row Scales Array On The Gpu
        CudaSetRowScales(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(EffNbrColumns, FoldingMaxLength=2048) / 2.0)), 1, 1), grid=(NbrRows, 1))  # noqa
        CudaRescaleRows(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(EffNbrColumns))), 1, 1), grid=(NbrRows, 1))  # noqa

    time_on_gpu, time_pivoting, time_compare, time_rescale, time_reduce, time_increment = [], [], [], [], [], []

    # Reduction To Reduced Echelon Form
    for i in range(NbrColumns):

        if verbose:  # perhaps limit printing rate
            print(f"\r@{i}/{max(NbrRows, NbrColumns)}               ", end="")

        time_on_gpu += [-time.time()]

        time_pivoting += [-time.time()]
        # Scaled Partial Pivoting - for finite fields partial rook pivoting would be equally good
        CudaThreadsReduceToMaxIndex(matrix_gpu,  # noqa
                                    block=(int(math.ceil(folded_number_of_columns(NbrRows, FoldingMaxLength=2048) / 2.0)), 1, 1),
                                    grid=(number_of_foldings(NbrRows, FoldingMaxLength=2048), 1))
        if number_of_foldings(NbrRows, FoldingMaxLength=2048) > 1:
            CudaBlocksReduceToMaxIndex(matrix_gpu, block=(int(math.ceil(number_of_foldings(NbrRows, FoldingMaxLength=2048) / 2)), 1, 1), grid=(1, 1))  # noqa
        CudaSwitchRows(matrix_gpu, block=(folded_number_of_columns(NbrColumns), 1, 1), grid=(number_of_foldings(NbrColumns), 1))  # noqa
        time_pivoting[-1] += time.time()

        time_compare += [-time.time()]
        CudaCompareHeadToTolerance(matrix_gpu, block=(1, 1, 1), grid=(1, 1))  # noqa
        time_compare[-1] += time.time()

        time_rescale += [-time.time()]
        CudaConditionalRescaleRow(matrix_gpu, block=(folded_number_of_columns(NbrColumns), 1, 1), grid=(number_of_foldings(NbrColumns), 1))  # noqa
        time_rescale[-1] += time.time()

        if i == NbrColumns:
            break

        time_reduce += [-time.time()]
        CudaConditionalRowReduce(matrix_gpu, block=(round_to_multiple_of(folded_number_of_columns(EffNbrColumns - i, 512), 32), 1, 1), grid=(NbrRows, 1))  # noqa
        time_reduce[-1] += time.time()

        time_increment += [-time.time()]
        # Increment Mirrored Counters
        CudaIncrementCounters(block=(1, 1, 1), grid=(1, 1))  # noqa
        time_increment[-1] += time.time()

        time_on_gpu[-1] += time.time()

    if verbose:
        print("\rTime elapsed on gpu: ", sum(time_on_gpu), ". ", end="\n")
        print("time_pivoting", sum(time_pivoting), end="\n")
        print("time_compare", sum(time_compare), end="\n")
        print("time_rescale", sum(time_rescale), end="\n")
        print("time_reduce", sum(time_reduce), end="\n")
        print("time_increment", sum(time_increment), end="\n")

    # Pull Matrix From Device
    matrix_copier = cuda.Memcpy2D()
    matrix_copier.set_src_device(matrix_gpu)
    matrix_copier.set_dst_host(matrix_cpu)
    matrix_copier.width_in_bytes = width
    matrix_copier.src_pitch = pitch
    matrix_copier.dst_pitch = width
    matrix_copier.height = height
    matrix_copier(aligned=True)

    return matrix_cpu

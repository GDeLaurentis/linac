# Author: Giuseppe
# Created: 11/02/2021

import os
import math
import time

from linac.pycuda_tools import cuda_set_vars_and_get_funcs, number_of_foldings, folded_number_of_columns, round_to_multiple_of
from linac.timeit_decorator import timeit

local_directory = os.path.dirname(os.path.abspath(__file__))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@timeit
def cuda_row_reduce(matrix, field_characteristic=0, verbose=False):

    import pycuda.driver as cuda
    import pycuda.autoinit                     # noqa
    from pycuda.compiler import SourceModule   # noqa

    # Compile Cuda Code - sets Cuda* Functions
    NbrRows, NbrColumns = matrix.shape

    # Push Matrix To Device
    if field_characteristic == 0:  # runs with complex128
        matrix_cpu = matrix.astype("complex128")
    else:  # runs with unsigned int (32 bits)
        matrix_cpu = matrix.astype('uint32')
    width = NbrColumns * matrix_cpu.dtype.itemsize
    height = NbrRows

    # Need alignment in order to access uint32 as 4 vector
    access_size = 16
    chunksize = access_size // matrix_cpu.dtype.itemsize
    if width%access_size == 0:
        pad = 0
    else:
        pad = access_size - width%access_size
    pitch = width + pad
    matrix_gpu = cuda.mem_alloc(pitch*height)

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
                                         NBR_ROWS=NbrRows, NBR_COLUMNS=EffNbrColumns, FIELD_CHARACTERISTIC=field_characteristic, )), locals(), globals())

    if field_characteristic == 0:  # Set The Row Scales Array On The Gpu
        CudaSetRowScales(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(NbrColumns, FoldingMaxLength=2048) / 2.0)), 1, 1), grid=(NbrRows, 1))  # noqa
        CudaRescaleRows(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(NbrColumns))), 1, 1), grid=(NbrRows, 1))  # noqa

    time_on_gpu, time_pivoting, time_compare, time_rescale, time_reduce, time_increment = [], [], [], [], [], []

    # Reduction To Reduced Echelon Form
    for i in range(max(NbrRows, NbrColumns)):

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
        CudaCompareHeadToTollerance(matrix_gpu, block=(1, 1, 1), grid=(1, 1))  # noqa
        time_compare[-1] += time.time()

        time_rescale += [-time.time()]
        CudaConditionalRescaleRow(matrix_gpu, block=(folded_number_of_columns(NbrColumns), 1, 1), grid=(number_of_foldings(NbrColumns), 1))  # noqa
        time_rescale[-1] += time.time()

        time_reduce += [-time.time()]
        CudaConditionalRowReduce(matrix_gpu, block=(round_to_multiple_of(folded_number_of_columns(EffNbrColumns - i, 512), 32), 1, 1), grid=(NbrRows, 1))  # noqa
        time_reduce[-1] += time.time()

        time_increment += [-time.time()]
        # Increment Mirrored Counters
        CudaIncrementCounters(block=(1, 1, 1), grid=(1, 1))  # noqa
        time_increment[-1] += time.time()

        time_on_gpu[-1] += time.time()

        # if i % 100 == 0:
        #        time.sleep(0.5)

    if verbose:
        print("\rTime elapsed on gpu: ", sum(time_on_gpu), ". ", end="\n")
        if field_characteristic == 0:
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

    print("NThreads = {}".format(folded_number_of_columns(EffNbrColumns//chunksize)))
    print(matrix_cpu[-1,:])

    return matrix_cpu


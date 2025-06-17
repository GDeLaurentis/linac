# Author: Giuseppe
# Created: 11/02/2021

import os
import math
import time

from linac.pycuda_tools import cuda_set_vars_and_get_funcs, number_of_foldings, folded_number_of_columns
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

    exec(str(cuda_set_vars_and_get_funcs(path_to_cuda_script=local_directory + "/row_reduce.cu",
                                         NBR_ROWS=NbrRows, NBR_COLUMNS=NbrColumns, FIELD_CHARACTERISTIC=field_characteristic, )), locals(), globals())

    # Push Matrix To Device
    if field_characteristic == 0:  # runs with complex128
        matrix_cpu = matrix.astype("complex128")
    else:  # runs with unsigned int (32 bits)
        matrix_cpu = matrix.astype('uint32')
    matrix_gpu = cuda.mem_alloc(matrix_cpu.size * matrix_cpu.dtype.itemsize)
    cuda.memcpy_htod(matrix_gpu, matrix_cpu)

    if field_characteristic == 0:  # Set The Row Scales Array On The Gpu
        CudaSetRowScales(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(NbrColumns, FoldingMaxLength=2048) / 2.0)), 1, 1), grid=(NbrRows, 1))  # noqa
        CudaRescaleRows(matrix_gpu, block=(int(math.ceil(folded_number_of_columns(NbrColumns))), 1, 1), grid=(NbrRows, 1))  # noqa

    time_on_gpu, time_pivoting, time_compare, time_rescale, time_reduce, time_increment = [], [], [], [], [], []

    # Reduction To Reduced Echelon Form
    for i in range(max(NbrRows, NbrColumns)):

        if verbose:
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
        CudaCompareHeadToTollerance(matrix_gpu, block=(1, 1, 1), grid=(1, 1))  # noqa
        time_compare[-1] += time.time()

        time_rescale += [-time.time()]
        CudaConditionalRescaleRow(matrix_gpu, block=(folded_number_of_columns(NbrColumns), 1, 1), grid=(number_of_foldings(NbrColumns), 1))  # noqa
        time_rescale[-1] += time.time()

        time_reduce += [-time.time()]
        CudaConditionalRowReduce(matrix_gpu, block=(folded_number_of_columns(NbrColumns), 1, 1), grid=(NbrRows, 1))  # noqa
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

    # Pull Matbrix From Device
    cuda.memcpy_dtoh(matrix_cpu, matrix_gpu)

    return matrix_cpu

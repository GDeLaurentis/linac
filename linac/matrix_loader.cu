#define FIELD_CHARACTERISTIC {FIELD_CHARACTERISTIC}
#define NBR_ROWS {NBR_ROWS}
#define NBR_COLUMNS {NBR_COLUMNS}
#define BASIS_LENGTH {BASIS_LENGTH}
#define DEGREE {DEGREE}

#include <pycuda-complex.hpp>
#include <stdio.h>
#include <math.h>

#if FIELD_CHARACTERISTIC > 0
typedef unsigned int matrix_type;
#else
typedef pycuda::complex<double> matrix_type;
#endif


/*!!!  DECLARATION  !!!*/

// DEVICE VARIABLES

__device__ int NbrRows = NBR_ROWS;
__device__ int NbrColumns = NBR_COLUMNS;
__device__ int MaxMatrixId = NBR_ROWS * NBR_COLUMNS;

#if FIELD_CHARACTERISTIC > 0
__device__ __constant__ unsigned int prime = FIELD_CHARACTERISTIC;
#endif

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ unsigned int Product64 (unsigned long int a, unsigned long int b);
#endif

// GLOBAL FUNCTIONS

__global__ void LoadMatrix (matrix_type *matrix, matrix_type *bases, int *indices);


/*!!!  IMPLEMENTATION  !!!*/

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ unsigned int Product64 (unsigned long int a, unsigned long int b) {
    return (a * b) % prime;
}
#endif

// GLOBAL FUNCTIONS

__global__ void LoadMatrix (matrix_type *matrix, matrix_type *bases, int *indices) {
    __shared__ matrix_type basis[BASIS_LENGTH];

    int FoldingLength = blockDim.x;
    int NbrFoldings = ceil(NbrColumns / (1.0 * FoldingLength));

    __syncthreads();

    for (int s = 0; s < ceil(double(BASIS_LENGTH)/blockDim.x); s++) {
        int basis_index = s * blockDim.x + threadIdx.x;
        if (basis_index < BASIS_LENGTH) {
            basis[basis_index] = bases[blockIdx.x * BASIS_LENGTH + basis_index];
        }
    }

    __syncthreads();

    for (int s = 0; s < NbrFoldings; s++) {
        int column_id = threadIdx.x + s * FoldingLength;
        if (column_id < NbrColumns) {
            matrix[blockIdx.x * NbrColumns + column_id] = basis[indices[column_id * DEGREE]];
            for (int t = 1; t < DEGREE; t++) {
#if FIELD_CHARACTERISTIC > 0
            matrix[blockIdx.x * NbrColumns + column_id] = Product64(matrix[blockIdx.x * NbrColumns + column_id], basis[indices[column_id * DEGREE + t]]);
#else
            matrix[blockIdx.x * NbrColumns + column_id] = matrix[blockIdx.x * NbrColumns + column_id] * basis[indices[column_id * DEGREE + t]];
#endif
            }
        }
    }
}

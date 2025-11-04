#define FIELD_CHARACTERISTIC {FIELD_CHARACTERISTIC}
#define NBR_ROWS {NBR_ROWS}
#define NBR_COLUMNS {NBR_COLUMNS}
#define BASIS_LENGTH {BASIS_LENGTH}
#define DEGREE {DEGREE}

#include <pycuda-complex.hpp>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>

#if FIELD_CHARACTERISTIC > 0
typedef unsigned int matrix_type;
#else
typedef cuDoubleComplex matrix_type;
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
__device__ unsigned int mul (unsigned long int a, unsigned long int b);
#else
__device__ inline cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b);
#endif

// GLOBAL FUNCTIONS

__global__ void LoadMatrix (matrix_type *matrix, matrix_type *bases, int *indices);


/*!!!  IMPLEMENTATION  !!!*/

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ inline unsigned int mul (unsigned long int a, unsigned long int b) {
    return (a * b) % prime;
}
#else
__device__ inline cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
#endif

// GLOBAL FUNCTIONS

__global__ void LoadMatrix (matrix_type *matrix, matrix_type *bases, int *indices) {
    __shared__ matrix_type basis[BASIS_LENGTH];

    int FoldingLength = blockDim.x;
    int NbrFoldings = ceil(NbrColumns / (1.0 * FoldingLength));

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
                matrix[blockIdx.x * NbrColumns + column_id] = mul(matrix[blockIdx.x * NbrColumns + column_id], basis[indices[column_id * DEGREE + t]]);
            }
        }
    }
}

#define FIELD_CHARACTERISTIC {FIELD_CHARACTERISTIC}{EXTENSION}
#define REAL {REAL}
#define MOD64 {MOD64}
#define NBR_ROWS {NBR_ROWS}
#define NBR_COLUMNS {NBR_COLUMNS}
#define BASIS_LENGTH {BASIS_LENGTH}
#define DEGREE {DEGREE}

#include <pycuda-complex.hpp>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <stdint.h>

#if FIELD_CHARACTERISTIC > 0
  #if MOD64
    using matrix_type = uint64_t;
    using unsig_t     = uint64_t;
    using wide_t      = unsigned __int128;
  #else
    using matrix_type = uint32_t;
    using unsig_t     = uint32_t;
    using wide_t      = uint64_t;
  #endif
#else
  #if REAL
    using matrix_type = double;          // real numbers (64-bit)
  #else
    using matrix_type = cuDoubleComplex; // complex numbers (2×64-bit)
  #endif
#endif


/*!!!  DECLARATION  !!!*/

// DEVICE VARIABLES

__device__ __constant__ unsigned long int NbrRows = NBR_ROWS;
__device__ __constant__ unsigned long int NbrColumns = NBR_COLUMNS;
__device__ __constant__ unsigned long int MaxMatrixId = NBR_ROWS * NBR_COLUMNS;

#if FIELD_CHARACTERISTIC > 0
__device__ __constant__ unsig_t prime = FIELD_CHARACTERISTIC;
#endif

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ __forceinline__ unsig_t mul(unsig_t a, unsig_t b);
#else
  #if REAL
__device__ __forceinline__ double mul(double a, double b);
  #else
__device__ __forceinline__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b);
  #endif
#endif

// GLOBAL FUNCTIONS

__global__ void LoadMatrix (matrix_type *matrix, matrix_type *bases, int *indices);


/*!!!  IMPLEMENTATION  !!!*/

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ __forceinline__ unsig_t mul(unsig_t a, unsig_t b) {
    wide_t prod = static_cast<wide_t>(a) * static_cast<wide_t>(b);
    return static_cast<unsig_t>(prod % static_cast<wide_t>(FIELD_CHARACTERISTIC));
}
#else
  #if REAL
__device__ __forceinline__ double mul(double a, double b) {
    return a * b;
}
  #else
__device__ __forceinline__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
  #endif
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

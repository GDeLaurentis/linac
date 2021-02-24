#define FIELD_CHARACTERISTIC {FIELD_CHARACTERISTIC}
#define NBR_ROWS {NBR_ROWS}
#define NBR_COLUMNS {NBR_COLUMNS}

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
__device__ int i = 0;  // row counter
__device__ int j = 0;  // column counter
__device__ bool bHeadIsBiggerThanTollerance = true;

#if FIELD_CHARACTERISTIC > 0
__device__ __constant__ unsigned int prime = FIELD_CHARACTERISTIC;
__device__ int tollerance = 0;
#else
__device__ double tollerance = 0.000000001; // 10^-9
__device__ double RowScales[8192]; // 2^13
__device__ int IndexOfMaximum = 0;
__device__ int IndicesOfMaxiumCandidates[16];
#endif

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ int Inverse (int a);
__device__ unsigned int ModP (int a);
__device__ unsigned int Product64 (unsigned long int a, unsigned long int b);
#endif
__device__ void RescaleRow (matrix_type *Matrix);
__device__ void RowReduce (matrix_type *Matrix);

// GLOBAL FUNCTIONS

__global__ void IncrementCounters ();
__global__ void CompareHeadToTollerance (matrix_type *Matrix);
__global__ void ConditionalRescaleRow (matrix_type *Matri);
__global__ void ConditionalRowReduce (matrix_type *Matrix);
#if FIELD_CHARACTERISTIC == 0
__global__ void SetRowScales (matrix_type *Matrix);
__global__ void RescaleRows (matrix_type *Matrix);
__global__ void SwitchRows (matrix_type *Matrix);
__global__ void ThreadsReduceToMaxIndex (matrix_type *Matrix);
__global__ void BlocksReduceToMaxIndex (matrix_type *Matrix);
#else
#endif


/*!!!  IMPLEMENTATION  !!!*/

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ int Inverse (int a) {

    int quotient, old_old_r, old_old_s, old_old_t;
    // int gcd;
    
    int b = prime;
    
    int old_r = a;
    int r = b;
    int old_s = 1;
    int s = 0;
    int old_t = 0;
    int t = 1;

    while (r != 0) {
	quotient = old_r / r;
	old_old_r = old_r;
	old_r = r;
	r = old_old_r - quotient * old_r;
	old_old_s = old_s;
	old_s = s;
	s = old_old_s - quotient * old_s;
	old_old_t = old_t;
	old_t = t;
	t = old_old_t - quotient * old_t;
    }

    // output "Bézout coefficients:", (old_s, old_t)
    // output "greatest common divisor:", old_r
    // output "quotients by the gcd:", (t, s)

    s = old_s;
    // t = old_t;
    // gcd = old_r;

    if (s > 0) {
	return s;
    } else {
	return s + prime;
    }
}

__device__ unsigned int Product64 (unsigned long int a, unsigned long int b) {
    return (a * b) % prime;
}

__device__ unsigned int ModP (int a) {
    if (a < 0) {
	return a + prime;
    } else {
	return a;
    }
}
#endif

__device__ void RescaleRow(matrix_type *Matrix) {
    int FoldingLength = blockDim.x;
    int id_i_head = i * NbrColumns + j;
    int MaxId = (i + 1) * NbrColumns;
    int id = i * NbrColumns + FoldingLength * blockIdx.x + threadIdx.x;
    if (id < MaxId) {
#if FIELD_CHARACTERISTIC > 0
	Matrix[id] = Product64(Matrix[id], Inverse(Matrix[id_i_head]));
#else
	Matrix[id] = Matrix[id] / Matrix[id_i_head];
#endif
    }
}

__device__ void RowReduce(matrix_type *Matrix) {
    int FoldingLength = blockDim.x;
    int NbrFoldings = ceil(NbrColumns / (1.0 * FoldingLength));
    int id_j_head = blockIdx.x * NbrColumns + j;
    int idMax = (blockIdx.x + 1) * NbrColumns;
    for (int s = 0; s < NbrFoldings; s++) {
	int id = blockIdx.x * NbrColumns + s * FoldingLength + threadIdx.x;
	int id_i = i * NbrColumns + s * FoldingLength + threadIdx.x;
	if (blockIdx.x != i && s * FoldingLength + threadIdx.x > j && id < MaxMatrixId && id < idMax){
	    #if FIELD_CHARACTERISTIC > 0
	    Matrix[id] = ModP(Matrix[id] - Product64(Matrix[id_i], Matrix[id_j_head]));
	    #else
	    Matrix[id] = Matrix[id] - Matrix[id_i] * Matrix[id_j_head];
	    #endif
	}
    }
    __syncthreads();
    if (blockIdx.x != i && threadIdx.x == 0 && id_j_head < MaxMatrixId) {
	Matrix[id_j_head] = 0;
    }
}

// GLOBAL FUNCTIONS

#if FIELD_CHARACTERISTIC == 0
__global__ void SetRowScales(matrix_type *Matrix) {
    __shared__ double sdata[4096];

    int RowId = blockIdx.x;
    int NbrFoldings = ceil(NbrColumns / (1.0 * blockDim.x) / 2.);
    int NbrColumnsPerIteration = ceil(NbrColumns / (1.0 * NbrFoldings));

    for (unsigned int r = 0; r < NbrFoldings; r++) {
	int sdataId = threadIdx.x + r * blockDim.x;
	int ColumnId = threadIdx.x + r * NbrColumnsPerIteration;
	int MatrixId1 = RowId * NbrColumns + ColumnId;
	int MatrixId2 = RowId * NbrColumns + ColumnId + blockDim.x;
	if (MatrixId2 < (RowId + 1) * NbrColumns) {
	    double value1 = abs(Matrix[MatrixId1]);
	    double value2 = abs(Matrix[MatrixId2]);
	    if (value1 > value2) {
		sdata[sdataId] = value1;
	    } else {
		sdata[sdataId] = value2;
	    }
	} else { // This is for odd lengths: the last one doesn't have something to compare it to.
	    sdata[sdataId] = abs(Matrix[MatrixId1]);
	}
    }
    __syncthreads();

    for (unsigned int r = 0; r < NbrFoldings; r++) {
	int sdataId = threadIdx.x + r * blockDim.x;
	for (unsigned int s = (blockDim.x + 1) / 2; s > 0; (s == 1) ? s = 0 : s = s+1 >> 1) {
	    if ((sdataId < s + r * blockDim.x) && ((sdataId + s) <= NbrColumns / 2) && sdata[sdataId] < sdata[sdataId + s]) {
		sdata[sdataId] = sdata[sdataId + s];
	    }
	    __syncthreads();
	}
    }

    for (unsigned int s = (NbrFoldings + 1) / 2; s > 0; (s == 1) ? s = 0 : s = s+1 >> 1) {
	if (threadIdx.x < s) {
	    int sdataId = 0 + s * threadIdx.x * blockDim.x;
	    if (sdata[sdataId] < sdata[sdataId + blockDim.x]) {
		sdata[sdataId] = sdata[sdataId + blockDim.x];
	    }
	}
    }

    if (threadIdx.x == 0) {
	RowScales[blockIdx.x] = sdata[0];
    }
}

__global__ void RescaleRows (matrix_type *Matrix) {
    __shared__ double rowScale;

    if (threadIdx.x == 0) {
	rowScale = RowScales[blockIdx.x];
    }
    __syncthreads();

    int RowId = blockIdx.x;
    int NbrFoldings = ceil(NbrColumns / (1.0 * blockDim.x));
    int NbrColumnsPerIteration = ceil(NbrColumns / (1.0 * NbrFoldings));

    for (unsigned int r = 0; r < NbrFoldings; r++) {
	int ColumnId = threadIdx.x + r * NbrColumnsPerIteration;
	int MatrixId = RowId * NbrColumns + ColumnId;
	if (MatrixId < (RowId + 1) * NbrColumns) {
	    Matrix[MatrixId] = Matrix[MatrixId] / rowScale;
	}
    }
}

__global__ void SwitchRows(matrix_type *Matrix) {
    int FoldingLength = blockDim.x;
    int id_i = i * NbrColumns + blockIdx.x * FoldingLength + threadIdx.x;
    int id_j = IndexOfMaximum * NbrColumns + blockIdx.x * FoldingLength + threadIdx.x;
    if (id_j < (IndexOfMaximum + 1) * NbrColumns && id_i < MaxMatrixId && id_j < MaxMatrixId) {
	pycuda::complex<double> temporary = Matrix[id_i];
	Matrix[id_i] = Matrix[id_j];
	Matrix[id_j] = temporary;
    }
}

__global__ void ThreadsReduceToMaxIndex(matrix_type *Matrix) {
    __shared__ double sdata[1024];
    __shared__ int idata[1024];

    // blockDim is (len+1)/2
    int NbrFoldings = gridDim.x;
    int tid = threadIdx.x;
    int RowId1 = tid + i;                      // ColumnId is always j
    int RowId2 = RowId1 + blockDim.x;          // ColumnId is always j
    int MatrixId1 = RowId1 * NbrColumns + j;
    int MatrixId2 = RowId2 * NbrColumns + j;
    if (RowId2 < NbrRows) {
	double value1 = abs(Matrix[MatrixId1]);
	double value2 = abs(Matrix[MatrixId2]);
	if (value1 > value2){
	    idata[tid] = RowId1;
	    sdata[tid] = value1;
	} else {
	    idata[tid] = RowId2;
	    sdata[tid] = value2;
	}
    } else if (RowId1 < NbrRows) {
	idata[tid] = RowId1;
	sdata[tid] = abs(Matrix[MatrixId1]);
    }
    __syncthreads();

    for (unsigned int s = (blockDim.x + 1) / 2; s > 0; (s == 1) ? s = 0 : s = s+1 >> 1) {
	if ((tid < s) && ((tid + s) < blockDim.x) && (sdata[tid] < sdata[tid + s])){
	    idata[tid] = idata[tid + s];
	    sdata[tid] = sdata[tid + s];
	}
	__syncthreads();
    }

    if (tid == 0 && NbrFoldings == 1){
	IndexOfMaximum = idata[0];
    } else if (tid == 0) {
	IndicesOfMaxiumCandidates[blockIdx.x] = idata[0];
    }
}

__global__ void BlocksReduceToMaxIndex(matrix_type *Matrix) {
    __shared__ double sdata[8];
    __shared__ int idata[8];
    // int NbrCandidates = blockDim.x;
    int tid = threadIdx.x;
    int RowId1 = IndicesOfMaxiumCandidates[tid * 2];
    int RowId2 =  IndicesOfMaxiumCandidates[tid * 2 + 1];
    int MatrixId1 = RowId1 * NbrColumns + j;
    int MatrixId2 = RowId2 * NbrColumns + j;
    if (MatrixId2 < MaxMatrixId && RowId2 >= i) {
	double value1 = abs(Matrix[MatrixId1]);
	double value2 = abs(Matrix[MatrixId2]);
	if (value1 > value2){
	    idata[tid] = RowId1;
	    sdata[tid] = value1;
	} else {
	    idata[tid] = RowId2;
	    sdata[tid] = value2;
	}
    } else if (MatrixId1 < MaxMatrixId && RowId1 >= i) {
	sdata[tid] = abs(Matrix[MatrixId1]);
	idata[tid] = RowId1;
    }

    for (unsigned int s = (blockDim.x + 1) / 2; s > 0; (s == 1) ? s = 0 : s = s+1 >> 1) {
	if ((tid < s) && ((tid + s) < blockDim.x) && (sdata[tid] < sdata[tid + s])){
	    idata[tid] = idata[tid + s];
	    sdata[tid] = sdata[tid + s];
	}
	__syncthreads();
    }

    if (tid == 0){
	IndexOfMaximum = idata[tid];
    }
  
}
#endif

__global__ void IncrementCounters () {
    if (bHeadIsBiggerThanTollerance) {
	i += 1;
	j += 1;
    } else {
	j += 1;
    }
}

__global__ void CompareHeadToTollerance(matrix_type *Matrix) {
    int MatrixId = i * NbrColumns + j;
#if FIELD_CHARACTERISTIC > 0
    if (MatrixId < MaxMatrixId && Matrix[MatrixId] != 0) {
#else
    if (MatrixId < MaxMatrixId && abs(Matrix[MatrixId]) > tollerance) {
#endif
	bHeadIsBiggerThanTollerance = true;
    } else {
	bHeadIsBiggerThanTollerance = false;
    }
}

__global__ void ConditionalRescaleRow(matrix_type *Matrix) {
    if (bHeadIsBiggerThanTollerance == true){
	RescaleRow(Matrix);
    }
}

__global__ void ConditionalRowReduce(matrix_type *Matrix) {
    if (bHeadIsBiggerThanTollerance == true){
	RowReduce(Matrix);
    }
}

#define FIELD_CHARACTERISTIC {FIELD_CHARACTERISTIC}{EXTENSION}
#define REAL {REAL}
#define MOD64 {MOD64}
#define NBR_ROWS {NBR_ROWS}
#define NBR_COLUMNS {NBR_COLUMNS}  // Pitch - includes padding
#define TRUE_NBR_COLUMNS {TRUE_NBR_COLUMNS}  // True (logical) number of columns

#include <pycuda-complex.hpp>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#if FIELD_CHARACTERISTIC > 0
  #if MOD64
    using matrix_type = uint64_t;
    using unsig_t     = uint64_t;
    using sig_t       = int64_t;
    using wide_t      = unsigned __int128;
    using pack_t      = ulonglong2;
    #define CHUNKSIZE 2
  #else
    using matrix_type = uint32_t;
    using unsig_t     = uint32_t;
    using sig_t       = int32_t;
    using wide_t      = uint64_t;
    using pack_t      = uint4;
    #define CHUNKSIZE 4
  #endif
#else
  #if REAL
    using matrix_type = double;                  // real numbers (64-bit)
    using pack_t      = double2;
    #define CHUNKSIZE 2
  #else
    using matrix_type = pycuda::complex<double>; // complex numbers (2×64-bit)
    #define CHUNKSIZE 1
  #endif
#endif

/*!!!  DECLARATION  !!!*/

// DEVICE VARIABLES

__device__ __constant__ unsigned long int NbrRows = static_cast<unsigned long int>(NBR_ROWS);
__device__ __constant__ unsigned long int NbrColumns = static_cast<unsigned long int>(NBR_COLUMNS);
__device__ __constant__ unsigned long int TrueNbrColumns = static_cast<unsigned long int>(TRUE_NBR_COLUMNS);
__device__ __constant__ unsigned long int MaxMatrixId = static_cast<unsigned long int>(NBR_ROWS) * static_cast<unsigned long int>(NBR_COLUMNS);
__device__ unsigned long int i = 0;  // row counter
__device__ unsigned long int j = 0;  // column counter
__device__ bool bHeadIsBiggerThanTollerance = true;

#if FIELD_CHARACTERISTIC > 0
__device__ __constant__ unsig_t prime = FIELD_CHARACTERISTIC;
__device__ int tollerance = 0;
#else
__device__ double tollerance = 0.000000001; // 10^-9
__device__ double RowScales[8192]; // 2^13
#endif
__device__ unsigned int IndexOfMaximum = 0;
__device__ unsigned int IndicesOfMaxmiumCandidates[128];

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ __forceinline__ unsig_t Inverse(unsig_t a);
__device__ __forceinline__ unsig_t modp(sig_t a);
__device__ __forceinline__ unsig_t product_mod(unsig_t a, unsig_t b) ;
#endif
__device__ void RescaleRow (matrix_type *Matrix);
__device__ void RowReduce (matrix_type *Matrix);

// GLOBAL FUNCTIONS

__global__ void IncrementCounters ();
__global__ void CompareHeadToTollerance (matrix_type *Matrix);
__global__ void ConditionalRescaleRow (matrix_type *Matrix);
__global__ void ConditionalRowReduce (matrix_type *Matrix);
__global__ void SwitchRows (matrix_type *Matrix);
__global__ void ThreadsReduceToMaxIndex (matrix_type *Matrix);
__global__ void BlocksReduceToMaxIndex (matrix_type *Matrix);
#if FIELD_CHARACTERISTIC == 0
__global__ void SetRowScales (matrix_type *Matrix);
__global__ void RescaleRows (matrix_type *Matrix);
#endif


/*!!!  IMPLEMENTATION  !!!*/

// DEVICE FUNCTIONS

#if FIELD_CHARACTERISTIC > 0
__device__ __forceinline__ unsig_t Inverse(unsig_t a) {
    // Extended Euclid with signed temporaries, but unsigned IO.
    // Assumes 0 < a < prime and prime fits in unsig_t.
    sig_t old_r = static_cast<sig_t>(a), r = static_cast<sig_t>(prime);
    sig_t old_s = 1, s = 0;

    while (r != 0) {
        sig_t q = old_r / r;
        sig_t tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s;       old_s = s; s = tmp;
    }
    // old_s is the inverse modulo prime, normalize to [0, prime)
    if (old_s < 0) old_s += static_cast<sig_t>(prime);
    return static_cast<unsig_t>(old_s);
}

__device__ __forceinline__ unsig_t product_mod(unsig_t a, unsig_t b) {
    wide_t prod = static_cast<wide_t>(a) * static_cast<wide_t>(b);
    return static_cast<unsig_t>(prod % static_cast<wide_t>(FIELD_CHARACTERISTIC));
}

__device__ __forceinline__ unsig_t modp(sig_t a) {
    if (a < 0) {
        return static_cast<unsig_t>(a + static_cast<sig_t>(prime));
    } else {
        return static_cast<unsig_t>(a);
    }
}
#endif

__device__ void RescaleRow(matrix_type *Matrix) {
    int FoldingLength = blockDim.x;
    unsigned long int id_i_head = i * NbrColumns + j;
    unsigned long int MaxId     = i * NbrColumns + TrueNbrColumns;
    unsigned long int id        = i * NbrColumns + FoldingLength * blockIdx.x + threadIdx.x;
    if (id < MaxId) {
#if FIELD_CHARACTERISTIC > 0
        Matrix[id] = product_mod(Matrix[id], Inverse(Matrix[id_i_head]));
#else
        Matrix[id] = Matrix[id] / Matrix[id_i_head];
#endif
    }
}

__device__ void RowReduce(matrix_type *Matrix) {

    if (blockIdx.x == i) {
        return;
    }

    unsigned long int id_j_head = blockIdx.x * NbrColumns + j;
    matrix_type matrix_id_j_head = Matrix[id_j_head];

#if FIELD_CHARACTERISTIC > 0
    if (matrix_id_j_head == 0) {
        return;
    }
#endif

    int idx_min = j/CHUNKSIZE + threadIdx.x; 
    for (int idx = idx_min; idx < NbrColumns/CHUNKSIZE; idx += blockDim.x) {
        unsigned long int id = blockIdx.x * NbrColumns/CHUNKSIZE + idx;
        unsigned long int id_i = i * NbrColumns/CHUNKSIZE + idx;
#if CHUNKSIZE > 1
        pack_t result = reinterpret_cast<pack_t*>(Matrix)[id];
        pack_t ref = reinterpret_cast<pack_t*>(Matrix)[id_i];
  #if FIELD_CHARACTERISTIC > 0
        if(idx * CHUNKSIZE > j){
            result.x = modp(result.x - product_mod(ref.x, matrix_id_j_head));
        }
        if(idx * CHUNKSIZE + 1 > j){
            result.y = modp(result.y - product_mod(ref.y, matrix_id_j_head));
        }
    #if !MOD64
        if(idx * CHUNKSIZE + 2 > j){
            result.z = modp(result.z - product_mod(ref.z, matrix_id_j_head));
        }
        if(idx * CHUNKSIZE + 3 > j){
            result.w = modp(result.w - product_mod(ref.w, matrix_id_j_head));
        }
    #endif
  #elif REAL
        if(idx * CHUNKSIZE > j){
            result.x = result.x - ref.x * matrix_id_j_head;
        }
        if(idx * CHUNKSIZE + 1 > j){
            result.y = result.y - ref.y * matrix_id_j_head;
        }
  #endif
        reinterpret_cast<pack_t*>(Matrix)[id] = result;
#else
        if(idx > j){
            Matrix[id] = Matrix[id] - Matrix[id_i] * matrix_id_j_head;
        }
#endif
    }

    __syncthreads();

    if (threadIdx.x == 0 && id_j_head < MaxMatrixId) {
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
        if (MatrixId2 < RowId * NbrColumns + TrueNbrColumns) {
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
        if (MatrixId < RowId * NbrColumns + TrueNbrColumns) {
            Matrix[MatrixId] = Matrix[MatrixId] / rowScale;
        }
    }
}
#endif

__global__ void SwitchRows(matrix_type *Matrix) {
    int FoldingLength = blockDim.x;
    unsigned long int id_i = i * NbrColumns + blockIdx.x * FoldingLength + threadIdx.x;
    unsigned long int id_j = IndexOfMaximum * NbrColumns + blockIdx.x * FoldingLength + threadIdx.x;
    // if (threadIdx.x == 0) {
    // printf("Switching rows %d and %d. Max value: %i. \\n ", i, IndexOfMaximum, Matrix[id_j + j]);
    // }
    if (id_j < IndexOfMaximum * NbrColumns + TrueNbrColumns && id_i < MaxMatrixId && id_j < MaxMatrixId) {
        matrix_type temporary = Matrix[id_i];
        Matrix[id_i] = Matrix[id_j];
        Matrix[id_j] = temporary;
    }
}

__global__ void ThreadsReduceToMaxIndex(matrix_type *Matrix) {
#if FIELD_CHARACTERISTIC == 0
    __shared__ double sdata[1024];
#else
    __shared__ matrix_type sdata[1024];
#endif
    __shared__ int idata[1024];

    // blockDim is (len+1)/2
    
    int NbrFoldings = gridDim.x;
    int tid = threadIdx.x;
    sdata[tid] = 0;
    int RowId1 = tid + i;                      // ColumnId is always j
    int RowId2 = RowId1 + blockDim.x;          // ColumnId is always j
    unsigned long int MatrixId1 = RowId1 * NbrColumns + j;
    unsigned long int MatrixId2 = RowId2 * NbrColumns + j;
    if (RowId2 < NbrRows) {
#if FIELD_CHARACTERISTIC == 0
        double value1 = abs(Matrix[MatrixId1]);
        double value2 = abs(Matrix[MatrixId2]);
#else
        matrix_type value1 = Matrix[MatrixId1];
        matrix_type value2 = Matrix[MatrixId2];	
#endif
        if (value1 > value2){
            idata[tid] = RowId1;
            sdata[tid] = value1;
        } else {
            idata[tid] = RowId2;
            sdata[tid] = value2;
        }
    } else if (RowId1 < NbrRows) {
        idata[tid] = RowId1;
#if FIELD_CHARACTERISTIC == 0
        sdata[tid] = abs(Matrix[MatrixId1]);
#else
        sdata[tid] = Matrix[MatrixId1];
#endif
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
        IndicesOfMaxmiumCandidates[blockIdx.x] = idata[0];
    }
}

__global__ void BlocksReduceToMaxIndex(matrix_type *Matrix) {
#if FIELD_CHARACTERISTIC == 0
    __shared__ double sdata[256];
#else
    __shared__ matrix_type sdata[256];
#endif
    __shared__ int idata[256];
    // int NbrCandidates = blockDim.x;
    int tid = threadIdx.x;
    int RowId1 = IndicesOfMaxmiumCandidates[tid * 2];
    int RowId2 =  IndicesOfMaxmiumCandidates[tid * 2 + 1];
    unsigned long int MatrixId1 = RowId1 * NbrColumns + j;
    unsigned long int MatrixId2 = RowId2 * NbrColumns + j;
    if (MatrixId2 < MaxMatrixId && RowId2 >= i) {
#if FIELD_CHARACTERISTIC == 0
        double value1 = abs(Matrix[MatrixId1]);
        double value2 = abs(Matrix[MatrixId2]);
#else
        matrix_type value1 = Matrix[MatrixId1];
        matrix_type value2 = Matrix[MatrixId2];
#endif
        if (value1 > value2){
            idata[tid] = RowId1;
            sdata[tid] = value1;
        } else {
            idata[tid] = RowId2;
            sdata[tid] = value2;
        }
    } else if (MatrixId1 < MaxMatrixId && RowId1 >= i) {
        idata[tid] = RowId1;
#if FIELD_CHARACTERISTIC == 0
        sdata[tid] = abs(Matrix[MatrixId1]);
#else
        sdata[tid] = Matrix[MatrixId1];
#endif
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

__global__ void IncrementCounters () {
    if (bHeadIsBiggerThanTollerance) {
        i += 1;
        j += 1;
    } else {
        j += 1;
    }
}

__global__ void CompareHeadToTollerance(matrix_type *Matrix) {
    unsigned long int MatrixId = i * NbrColumns + j;
    // printf("Head MatrixId %li. \\n ", MatrixId);
    // printf("Head MaxMatrixId %li. \\n ", MaxMatrixId);
    // printf("i, j %li, %li. \\n ", i, j);
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

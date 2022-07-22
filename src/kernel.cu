/**
 * @file kernel.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "kernel.h"


template <typename T>
__global__ void cuCV::kernel::add(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    // threadIdx.x contains the index of the thread within the block
    // blockDim.x contains the size of thread block (number of threads in the thread block).

    // blockIdx.x contains the index of the block within the grid
    // gridDim.x contains the size of the grid

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] + B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::add(KernelCuMat<T> OUT, const KernelCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] + alpha;
}


template <typename T>
__global__ void cuCV::kernel::dif(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] - B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::dif(KernelCuMat<T> OUT, const KernelCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] - alpha;
}


template <typename T>
__global__ void cuCV::kernel::mul(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] * B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::mul(KernelCuMat<T> OUT, const KernelCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] * alpha;
}


template <typename T>
__global__ void cuCV::kernel::div(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] / B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::div(KernelCuMat<T> OUT, const KernelCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] / alpha;
}


template <typename T>
__global__ void cuCV::kernel::naiveMatmul(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels) {
        int OUTval = 0;

        for (int n = 0; n < A.mWidth; n++)  // Matrix dimensions: MxN @ NxL
            OUTval += A.mData[(row * A.mWidth + n) + (A.mWidth*A.mHeight)*ch] * B.mData[(n * B.mWidth + col) + (B.mWidth*B.mHeight)*ch];
        
        OUT.mData[(row * OUT.mWidth + col) + (OUT.mWidth*OUT.mHeight)*ch] = OUTval;
    }
}


template <typename T>
__global__ void cuCV::kernel::matmul(KernelCuMat<T> OUT, const KernelCuMat<T> A, const KernelCuMat<T> B) {
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int blockCh  = blockIdx.z;

    // Thread row and column within OUTsub
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Each thread block computes one sub-matrix OUTsub of OUT
    KernelCuMat<T> OUTsub = OUT.getSubCuMat(blockRow, blockCol, blockCh);

    // accumulate results in OUTval to set C_sub element
    T OUTval = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int n = 0; n < ((A.mWidth + BLOCK_SIZE - 1) / BLOCK_SIZE); n++) { // Matrix dimensions: MxN @ NxL
        // Get sub matrices of A and B
        KernelCuMat<T> Asub = A.getSubCuMat(blockRow, n);
        KernelCuMat<T> Bsub = B.getSubCuMat(n, blockCol);

        // Shared memory used to store elements of Asub und Bsub
        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Some entries in that matrix might be incorrect since the last submatrix might geht out of bounds of the full Matrix. Set those elems to 0.
        if (n * BLOCK_SIZE + threadCol < A.mWidth && blockRow * BLOCK_SIZE + threadRow < A.mHeight) {
            As[threadRow][threadCol] = Asub.getElement(threadRow, threadCol);
        } else {
            As[threadRow][threadCol] = 0.0;
        }
        if (n * BLOCK_SIZE + threadRow < B.mHeight && blockCol * BLOCK_SIZE + threadCol < B.mWidth) {
            Bs[threadRow][threadCol] = Bsub.getElement(threadRow, threadCol);
        } else {
            Bs[threadRow][threadCol] = 0.0;
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();    

        // Multiply Asub und Bsub together
        for (int elem = 0; (elem < BLOCK_SIZE); ++elem) {
            OUTval += As[threadRow][elem] * Bs[elem][threadCol];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    if (((blockCol * BLOCK_SIZE + threadCol) < OUT.mWidth) && ((blockRow * BLOCK_SIZE + threadRow) < OUT.mHeight)) {
        OUTsub.setElement(threadRow, threadCol, OUTval);
        //printf("Set (%d, %d, %d), stride: %d = %d ==/!= %d\n", threadRow, threadCol, threadIdx.z, OUT.mStride, OUTval, OUTsub.getElement(threadRow, threadCol));
    }
}


/// Explicit template specialization
template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::add(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::dif(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::mul(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::div(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::naiveMatmul(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::naiveMatmul(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::naiveMatmul(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::matmul(KernelCuMat<CUCV_8U> OUT, const KernelCuMat<CUCV_8U> A, const KernelCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::matmul(KernelCuMat<CUCV_16U> OUT, const KernelCuMat<CUCV_16U> A, const KernelCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::matmul(KernelCuMat<CUCV_64F> OUT, const KernelCuMat<CUCV_64F> A, const KernelCuMat<CUCV_64F> B);

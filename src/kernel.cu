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
__global__ void cuCV::kernel::add(CuMat<T> OUT, const CuMat<T> A, const CuMat<T> B) {
    // threadIdx.x contains the index of the thread within the block
    // blockDim.x contains the size of thread block (number of threads in the thread block).

    // blockIdx.x contains the index of the block within the grid
    // gridDim.x contains the size of the grid

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.mWidth + col;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight)
        OUT.mData[index] = A.mData[index] + B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::dif(CuMat<T> OUT, CuMat<T> A, CuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.mWidth + col;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight) {
        OUT.mData[index] = A.mData[index] - B.mData[index];
    }
}


template <typename T>
__global__ void cuCV::kernel::mul(CuMat<T> OUT, CuMat<T> A, CuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.mWidth + col;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight) {
        OUT.mData[index] = A.mData[index] - B.mData[index];
    }
}


template <typename T>
__global__ void cuCV::kernel::div(CuMat<T> OUT, CuMat<T> A, CuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.mWidth + col;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight) {
        OUT.mData[index] = A.mData[index] - B.mData[index];
    }
}


/// Explicit template specialization
template __global__ void cuCV::kernel::add(CuMat<CUCV_8U> OUT, const CuMat<CUCV_8U> A, const CuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::add(CuMat<CUCV_16U> OUT, const CuMat<CUCV_16U> A, const CuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::add(CuMat<CUCV_64F> OUT, const CuMat<CUCV_64F> A, const CuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::dif(CuMat<CUCV_8U> OUT, const CuMat<CUCV_8U> A, const CuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::dif(CuMat<CUCV_16U> OUT, const CuMat<CUCV_16U> A, const CuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::dif(CuMat<CUCV_64F> OUT, const CuMat<CUCV_64F> A, const CuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::mul(CuMat<CUCV_8U> OUT, const CuMat<CUCV_8U> A, const CuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::mul(CuMat<CUCV_16U> OUT, const CuMat<CUCV_16U> A, const CuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::mul(CuMat<CUCV_64F> OUT, const CuMat<CUCV_64F> A, const CuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::div(CuMat<CUCV_8U> OUT, const CuMat<CUCV_8U> A, const CuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::div(CuMat<CUCV_16U> OUT, const CuMat<CUCV_16U> A, const CuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::div(CuMat<CUCV_64F> OUT, const CuMat<CUCV_64F> A, const CuMat<CUCV_64F> B);


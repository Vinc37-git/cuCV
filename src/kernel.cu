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
        int OUT_value = 0;

        for (int n = 0; n < A.mWidth; n++)  // Matrix dimensions: MxN @ NxL
            OUT_value += A.mData[(row * A.mWidth + n) + (A.mWidth*A.mHeight)*ch] * B.mData[(n * B.mWidth + col) + (B.mWidth*B.mHeight)*ch];
        
        OUT.mData[(row * OUT.mWidth + col) + (OUT.mWidth*OUT.mHeight)*ch] = OUT_value;
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

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
__global__ void cuCV::kernel::add(CuMat<T> & OUT, CuMat<T> & A, CuMat<T> & B) {
    // threadIdx.x contains the index of the thread within the block
    // blockDim.x contains the size of thread block (number of threads in the thread block).

    // blockIdx.x contains the index of the block within the grid
    // gridDim.x contains the size of the grid

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.mWidth + col;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight) {
        OUT.mData[index] = A.mData[index] * B.mData[index];
    }
}


/*
template <typename T>
__global__ void cuCV::kernel::add(T * OUT) {
    //
}
*/
template <typename T>
__global__ void cuCV::kernel::dif<T>(cuCV::CuMat<T> OUT) {
    //
}

/// Explicit template specialization
template __global__ void cuCV::kernel::add(CuMat<unsigned char> & OUT, CuMat<unsigned char> & A, CuMat<unsigned char> & B);
template __global__ void cuCV::kernel::add(CuMat<unsigned short> & OUT, CuMat<unsigned short> & A, CuMat<unsigned short> & B);
template __global__ void cuCV::kernel::add(CuMat<double> & OUT, CuMat<double> & A, CuMat<double> & B);


//template __global__ void cuCV::kernel::add(unsigned char * OUT);
template __global__ void cuCV::kernel::dif<unsigned char>(cuCV::CuMat<unsigned char> OUT);
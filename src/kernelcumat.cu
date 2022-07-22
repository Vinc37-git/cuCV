/**
 * @file kernelcumat.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "kernelcumat.h"


// template <typename T>
// cuCV::KernelCuMat<T>::KernelCuMat() { }


// template <typename T>
// cuCV::KernelCuMat<T>::KernelCuMat(const cuCV::CuMat<T> & cuMat) 
//         : Mat<T>(cuMat.mWidth, cuMat.mHeight, cuMat.mChannels) {
//     this->mStride = cuMat.mStride;
//     this->mData = cuMat.mData;
// }


// template <typename T>
// cuCV::KernelCuMat<T>::KernelCuMat(const cuCV::KernelCuMat<T> & kernelCuMat) 
//         : Mat<T>(kernelCuMat.mWidth, kernelCuMat.mHeight, kernelCuMat.mChannels, kernelCuMat.mData) {
//     this->mStride = kernelCuMat.mStride;
// }

// template <typename T>
// cuCV::KernelCuMat<T>::~KernelCuMat() {
//     this->mData = NULL;
// } 


template <typename T> __host__ 
cuCV::KernelCuMat<T>::KernelCuMat(const cuCV::CuMat<T> & cuMat) 
        : mWidth(cuMat.mWidth), mHeight(cuMat.mHeight), mChannels(cuMat.mChannels), 
        mStride(cuMat.mStride),  mData(cuMat.mData) { }


template <typename T> __device__ 
cuCV::KernelCuMat<T>::KernelCuMat(int width, int height, int channels, int stride) 
        : mWidth(width), mHeight(height), mChannels(channels), 
        mStride(stride), mData(NULL) { }


template <typename T> __device__ 
cuCV::KernelCuMat<T> cuCV::KernelCuMat<T>::getSubCuMat(int blockIdRow, int blockIdCol, int blockIdCh) const {
    cuCV::KernelCuMat<T> sub(BLOCK_SIZE, BLOCK_SIZE, 1, mStride);
    sub.mData = & mData[mStride * blockIdRow * BLOCK_SIZE + blockIdCol * BLOCK_SIZE];
    return sub;
}


template <typename T> __device__ 
void cuCV::KernelCuMat<T>::setElement(const int row, const int col, const T value) {
    //printf("Set %d at %d with %d, %d, %d\n", value, mStride * row + col, mStride, row, col);
    mData[mStride * row + col] = value;
}


template <typename T> __device__  
T cuCV::KernelCuMat<T>::getElement(const int row, const int col) const {
    return mData[mStride * row + col];
}



/// Explicit template specialization
template class cuCV::KernelCuMat<CUCV_8U>;
template class cuCV::KernelCuMat<CUCV_16U>;
template class cuCV::KernelCuMat<CUCV_64F>;
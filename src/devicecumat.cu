/**
 * @file devicecumat.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "devicecumat.h"


template <typename T> __host__ 
cuCV::DeviceCuMat<T>::DeviceCuMat(const cuCV::CuMat<T> & cuMat) 
        : mWidth(cuMat.mWidth), mHeight(cuMat.mHeight), mChannels(cuMat.mChannels), 
        mStride(cuMat.mStride),  mData(cuMat.mData) { }


template <typename T> __device__ 
cuCV::DeviceCuMat<T>::DeviceCuMat(int width, int height, int channels, int stride) 
        : mWidth(width), mHeight(height), mChannels(channels), 
        mStride(stride), mData(NULL) { }


template <typename T> __device__ 
cuCV::DeviceCuMat<T> cuCV::DeviceCuMat<T>::getSubCuMat(int blockIdRow, int blockIdCol, int blockIdCh) const {
    cuCV::DeviceCuMat<T> sub(BLOCK_SIZE, BLOCK_SIZE, 1, mStride);
    sub.mData = & mData[mStride * blockIdRow * BLOCK_SIZE + blockIdCol * BLOCK_SIZE];
    return sub;
}


template <typename T> __device__ 
void cuCV::DeviceCuMat<T>::setElement(const int row, const int col, const T value) {
    //printf("Set %d at %d with %d, %d, %d\n", value, mStride * row + col, mStride, row, col);
    mData[mStride * row + col] = value;
}


template <typename T> __device__ 
void cuCV::DeviceCuMat<T>::setElement(const int row, const int col, const int ch, const T value) {
    //printf("Set %d at %d with %d, %d, %d\n", value, mStride * row + col, mStride, row, col);
    mData[mStride * row + col + ch * (mStride * mHeight)] = value;
    /// @bug add mStrideY and replace mHeight
}


template <typename T> __device__  
T cuCV::DeviceCuMat<T>::getElement(const int row, const int col) const {
    return mData[mStride * row + col];
}


template <typename T> __device__  
T cuCV::DeviceCuMat<T>::getElement(const int row, const int col, const int ch) const {
    return mData[mStride * row + col + (mStride*mHeight) * ch];
}



/// Explicit template specialization
template class cuCV::DeviceCuMat<CUCV_8U>;
template class cuCV::DeviceCuMat<CUCV_16U>;
template class cuCV::DeviceCuMat<CUCV_64F>;
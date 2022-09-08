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
        mStrideX(cuMat.mStrideX), mStrideY(cuMat.mStrideY), mData(cuMat.mData) { }


template <typename T> __device__ 
cuCV::DeviceCuMat<T>::DeviceCuMat(int width, int height, int channels, int strideX, int strideY) 
        : mWidth(width), mHeight(height), mChannels(channels), 
        mStrideX(strideX), mStrideY(strideY), mData(NULL) { }


template <typename T> __device__ 
cuCV::DeviceCuMat<T> cuCV::DeviceCuMat<T>::getBlock(int blockIdRow, int blockIdCol, int blockIdCh) const {
    cuCV::DeviceCuMat<T> sub(blockDim.x, blockDim.y, 1, mStrideX, mStrideY);
    sub.mData = & mData[mStrideX * blockIdRow * blockDim.y + blockIdCol * blockDim.x + mStrideX * mStrideY * blockIdCh];
    return sub;
}


template <typename T> __device__ 
cuCV::DeviceCuMat<T> cuCV::DeviceCuMat<T>::getSubCuMat(const int row, const int col, const int ch, const int width, const int height) const {
    cuCV::DeviceCuMat<T> sub(width, height, 1, mStrideX, mStrideY);
    sub.mData = & mData[mStrideX * row + col + mStrideX * mStrideY * ch];
    return sub;
}


template <typename T> __device__ 
void cuCV::DeviceCuMat<T>::setElement(const int row, const int col, const T value) {
    //printf("Set %d at %d with %d, %d, %d\n", value, mStrideX * row + col, mStrideX, row, col);
    mData[mStrideX * row + col] = value;
}


template <typename T> __device__ 
void cuCV::DeviceCuMat<T>::setElement(const int row, const int col, const int ch, const T value) {
    //printf("Set %d at %d with %d, %d, %d\n", value, mStrideX * row + col, mStrideX, row, col);
    mData[mStrideX * row + col + ch * (mStrideX * mStrideY)] = value;
}


template <typename T> __device__  
T cuCV::DeviceCuMat<T>::getElement(const int row, const int col) const {
    return mData[mStrideX * row + col];
}


template <typename T> __device__  
T cuCV::DeviceCuMat<T>::getElement(const int row, const int col, const int ch) const {
    return mData[mStrideX * row + col + (mStrideX * mStrideY) * ch];
}



/// Explicit template specialization
template class cuCV::DeviceCuMat<CUCV_8U>;
template class cuCV::DeviceCuMat<CUCV_16U>;
template class cuCV::DeviceCuMat<CUCV_32F>;
template class cuCV::DeviceCuMat<CUCV_64F>;
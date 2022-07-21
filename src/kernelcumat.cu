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


template <typename T>
cuCV::KernelCuMat<T>::KernelCuMat() { }


template <typename T>
cuCV::KernelCuMat<T>::KernelCuMat(const cuCV::CuMat<T> & cuMat) 
        : Mat<T>(cuMat.mWidth, cuMat.mHeight, cuMat.mChannels) {
    this->mStride = cuMat.mStride;
    this->mData = cuMat.mData;
}


template <typename T>
cuCV::KernelCuMat<T>::KernelCuMat(const cuCV::KernelCuMat<T> & kernelCuMat) 
        : Mat<T>(kernelCuMat.mWidth, kernelCuMat.mHeight, kernelCuMat.mChannels, kernelCuMat.mData) {
    this->mStride = kernelCuMat.mStride;
}

template <typename T>
cuCV::KernelCuMat<T>::~KernelCuMat() {
    this->mData = NULL;
} 


/// Explicit template specialization
template class cuCV::KernelCuMat<CUCV_8U>;
template class cuCV::KernelCuMat<CUCV_16U>;
template class cuCV::KernelCuMat<CUCV_64F>;
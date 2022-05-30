/**
 * @file cumat.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef CUMAT_CU
#define CUMAT_CU

#include "cumat.h"

template <typename T>
cuCV::CuMat<T>::CuMat() { }


template <typename T>
cuCV::CuMat<T>::CuMat(int width, int height, int channels, void * data) { }


template <typename T>
void cuCV::CuMat<T>::add(CuMat & OUT, CuMat & B) {

    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y);

    cuCV::kernel::add<<<blocks, threads>>>(OUT, *this, B);
}


template <typename T>
void cuCV::CuMat<T>::allocate(const Mat<T> & src) {
    if (this->mData != NULL) {
        /// @todo free
        std::cerr << "mData was not NULL" << std::endl;
        throw "mData was not NULL";
    }
    cudaMalloc(&this->mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels);
    /// @todo CUDA error handling 
}


template <typename T>
void cuCV::CuMat<T>::uploadFrom(const Mat<T> & src) {
    if (this->mData == NULL) {
        /// allocate
        allocate(src);
    }
    cudaMemcpy(this->mData, src.mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels, cudaMemcpyHostToDevice);
}


template <typename T>
void cuCV::CuMat<T>::downloadTo(const Mat<T> & dst) const {
    if (dst.mData == NULL) {
        std::cerr << "mData of destination is NULL" << std::endl;
        throw "mData of destination is NULL";
    }
    if (this->mData == NULL) {
        std::cerr << "mData of source on device is NULL" << std::endl;
        throw "mData of source on device is NULL";
    }
    cudaMemcpy(dst.mData, this->mData, sizeof(T) * dst.mWidth * dst.mHeight * dst.mChannels, cudaMemcpyDeviceToHost);
}


template <typename T>
void cuCV::CuMat<T>::free() {

}


/// Explicit template specialization
template class cuCV::CuMat<unsigned char>;
template class cuCV::CuMat<unsigned short>;
template class cuCV::CuMat<double>;

#endif //
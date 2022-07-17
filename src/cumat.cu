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


#include "cumat.h"

template <typename T>
cuCV::CuMat<T>::CuMat() { }


template <typename T>
cuCV::CuMat<T>::CuMat(Mat<T> & SRC) 
        : Mat<T>(SRC.mWidth, SRC.mHeight, SRC.mChannels, NULL) {  ///< We initialize Base Class Object "Mat" using parameters of SRC. However, dont point to the same data.
    allocateLike(SRC);
}


template <typename T>
void cuCV::CuMat<T>::add(CuMat & OUT, CuMat & B) {

    if (this->mData == NULL || OUT.mData == NULL || B.mData == NULL)
        throw std::invalid_argument("INVALID POINTERS DETECTED: pointers for addition are null.");

    if (this->mHeight != B.mHeight) {
        printf("DIMENSIONS DO NOT MATCH.");
        throw std::invalid_argument("DIMENSIONS DO NOT MATCH. @TODO CHANGE THAT");
    }

    printf("A: x=%d, y=%d, z=%d\n", this->mWidth, this->mHeight, this->mChannels);
    printf("B: x=%d, y=%d, z=%d\n", B.mWidth, B.mHeight, B.mChannels);
    printf("C: x=%d, y=%d, z=%d\n", OUT.mWidth, OUT.mHeight, OUT.mChannels);

    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y);

    cuCV::kernel::add<<<blocks, threads>>>(OUT, * this, B);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
bool cuCV::CuMat<T>::allocateLike(const Mat<T> & src) {
    if (this->mData != NULL) {
        /// @todo free
        throw std::bad_alloc(); // ("LEAK-WARNING: You are trying to allocate new VRAM but mData was not NULL")
    }

    /// Allocate Memory
    gpuErrchk(cudaMalloc((void**) & this->mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels));
    

    if (this->mData == NULL)
        return 0;
    else 
        return 1;
}


template <typename T>
void cuCV::CuMat<T>::uploadFrom(const Mat<T> & src) {
    /// Check wether 'this' contains already data. 
    /// If yes, the dimensions must match. If no, allocate, set dimensions and data.
    if (this->mData == NULL) {
        if (!allocateLike(src))
            throw std::bad_alloc(); //("Allocation of VRAM failed.")
        
        this->mWidth = src.mWidth;
        this->mHeight = src.mHeight;
        this->mChannels = src.mChannels;
        this->mStride = src.mStride;
    }
    else {  ///< Data is not null, so check dimensions.
        if (this->mWidth != src.mWidth || this->mHeight != src.mHeight ||
            this->mChannels != src.mChannels || this->mStride != src.mStride) 
            throw std::invalid_argument("DIMENSIONS DO NOT MATCH. @TODO CHANGE THAT");
        else {
            printf("'this' points to data. Please free data before allocating new data.");
            //abort();
        }
    }

    /// Send Memory from src to 'this'. 'this' is on device
    gpuErrchk(cudaMemcpy(this->mData, src.mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels, cudaMemcpyHostToDevice));   
}


template <typename T>
void cuCV::CuMat<T>::downloadTo(const Mat<T> & dst) const {
    if (dst.mData == NULL) {
        throw std::invalid_argument("mData of destination is NULL");  ///< @note undefined behaviour! 
    }
    if (this->mData == NULL) {
        throw std::invalid_argument("mData of source on device is NULL"); ///< @note undefined behaviour! 
    }
    gpuErrchk(cudaMemcpy(dst.mData, this->mData, sizeof(T) * dst.mWidth * dst.mHeight * dst.mChannels, cudaMemcpyDeviceToHost));
}


template <typename T>
void cuCV::CuMat<T>::free() {

}


/// Explicit template specialization
template class cuCV::CuMat<unsigned char>;
template class cuCV::CuMat<unsigned short>;
template class cuCV::CuMat<double>;
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
cuCV::CuMat<T>::CuMat(Mat<T> & mat) 
        : Mat<T>(mat.mWidth, mat.mHeight, mat.mChannels, NULL) {  ///< We initialize Base Class Object "Mat" using parameters of mat. However, dont point to the same data.
    //allocateLike(mat);
}


template <typename T>
cuCV::CuMat<T>::CuMat(int width, int height, int channels) 
        : Mat<T>(width, height, channels, NULL) { 
}


template <typename T>
cuCV::CuMat<T>::CuMat(const CuMat & cuMat)
        : Mat<T>(cuMat.mWidth, cuMat.mHeight, cuMat.mChannels, NULL) {
    if (cuMat.mData != NULL) {
        allocateLike(cuMat);
        gpuErrchk(cudaMemcpy(this->mData, cuMat.mData, sizeof(T) * cuMat.mWidth * cuMat.mHeight * cuMat.mChannels, cudaMemcpyDeviceToDevice)); 
    }
}


template <typename T>
cuCV::CuMat<T>::CuMat(CuMat && cuMat)
        : Mat<T>(cuMat.mWidth, cuMat.mHeight, cuMat.mChannels, cuMat.mData) {
    cuMat.mData = NULL;
}


template <typename T>
cuCV::CuMat<T>::~CuMat() {
    if (this->mData != NULL) {
        cudaFree(this->mData);
        this->mData = NULL;
    }
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator=(CuMat cuMat) {
    std::swap(this->mData, cuMat.mData);
    this->cuType = cuMat.cuType;
    this->mWidth = cuMat.mWidth; 
    this->mHeight = cuMat.mHeight; 
    this->mStride = cuMat.mStride;
    this->mChannels = cuMat.mChannels;
    return * this;
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator+=(const CuMat & cuMat) {
    if (this->mData == NULL || cuMat.mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (!compareDim(* this, cuMat))
        throw cuCV::exception::DimensionMismatch(* this, cuMat);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::add<<<blocks, threads>>>(this->kernel(), this->kernel(), cuMat.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator+=(T alpha) {
    if (this->mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::add<<<blocks, threads>>>(this->kernel(), this->kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator+(const CuMat & cuMat) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum += cuMat;
    return sum;
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator+(const T alpha) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum += alpha;
    return sum;
}



template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator-=(const CuMat & cuMat) {
    if (this->mData == NULL || cuMat.mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (!compareDim(* this, cuMat))
        throw cuCV::exception::DimensionMismatch(* this, cuMat);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::dif<<<blocks, threads>>>(this->kernel(), this->kernel(), cuMat.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator-=(T alpha) {
    if (this->mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::dif<<<blocks, threads>>>(this->kernel(), this->kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator-(const CuMat & cuMat) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum -= cuMat;
    return sum;
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator-(const T alpha) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum -= alpha;
    return sum;
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator*=(const CuMat & cuMat) {
    if (this->mData == NULL || cuMat.mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (!compareDim(* this, cuMat))
        throw cuCV::exception::DimensionMismatch(* this, cuMat);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::mul<<<blocks, threads>>>(this->kernel(), this->kernel(), cuMat.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator*=(T alpha) {
    if (this->mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::mul<<<blocks, threads>>>(this->kernel(), this->kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator*(const CuMat & cuMat) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum *= cuMat;
    return sum;
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator*(const T alpha) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum *= alpha;
    return sum;
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator/=(const CuMat & cuMat) {
    if (this->mData == NULL || cuMat.mData == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (!compareDim(* this, cuMat))
        throw cuCV::exception::DimensionMismatch(* this, cuMat);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::div<<<blocks, threads>>>(this->kernel(), this->kernel(), cuMat.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> & cuCV::CuMat<T>::operator/=(T alpha) {
    if (this->mData == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((this->mWidth + threads.x - 1) / threads.x, (this->mHeight + threads.y - 1) / threads.y, this->mChannels);

    /// Perform Math
    cuCV::kernel::div<<<blocks, threads>>>(this->kernel(), this->kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return * this;    
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator/(const CuMat & cuMat) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum /= cuMat;
    return sum;
}


template <typename T>
cuCV::CuMat<T> cuCV::CuMat<T>::operator/(const T alpha) const {
    cuCV::CuMat<T> sum(*this);  // Copy `this`
    sum /= alpha;
    return sum;
}


template <typename T>
void cuCV::CuMat<T>::uploadFrom(const Mat<T> & src) {
    /** Workflow
     * 2. Check dimensions. If uniintialised on device, initialise. If mismatch, throw error.
     * 3. Allocate MEM for device matrix. Dimensions checks makes sure count is right.
     * 4. Transfer data.
     */

    // Check if dimenions match. If dimensions are uninitialized assign them.
    if ((this->mWidth == 0) || (this->mHeight == 0) || (this->mChannels == 0)) {
        this->mWidth = src.mWidth;
        this->mHeight = src.mHeight;
        this->mChannels = src.mChannels;
        this->mStride = src.mStride;
    }
    else if (!compareDim(src, * this)) {
        throw cuCV::exception::DimensionMismatch(src, * this);
    }

    /// Allocate Memory on device in 'this' points to NULL. If it points to data, it will throw an exception.
    if (this->mData == NULL)
        allocateLike(src);

    /// Send Memory from src to 'this'. 'this' is on device
    gpuErrchk(cudaMemcpy(this->mData, src.mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels, cudaMemcpyHostToDevice));   
}


template <typename T>
void cuCV::CuMat<T>::downloadTo(Mat<T> & dst) const {
    /** Workflow
     * 1. Check if CuMat points to data
     * 2. Check dimensions. If uniintialised, initialised. If mismatch, throw error.
     * 3. Allocate MEM for host matrix. Dimensions checks makes sure count is right.
     * 4. Transfer data.
     */

    if (this->mData == NULL)
        throw cuCV::exception::NullPointer("Download failed. mData of source on device is NULL"); ///< @note undefined behaviour! 

    // Check if dimenions match. If dimensions are uninitialized assign them.
    if ((dst.mWidth == 0) || (dst.mHeight == 0) || (dst.mChannels == 0)) {
        dst.mWidth = this->mWidth;
        dst.mHeight = this->mHeight;
        dst.mChannels = this->mChannels;
        dst.mStride = this->mStride;
    }
    else if (!compareDim(* this, dst)) {
        throw cuCV::exception::DimensionMismatch(* this, dst);
    }

    // If MEM for destination on host is not allocated yet, allocate.
    if (dst.mData == NULL) {
        dst.alloc();  // The dim check makes sure the allocated size is always the right one for `this`.
    }

    gpuErrchk(cudaMemcpy(dst.mData, this->mData, sizeof(T) * dst.mWidth * dst.mHeight * dst.mChannels, cudaMemcpyDeviceToHost));
}


template <typename T>
void cuCV::CuMat<T>::allocateLike(const Mat<T> & src) {
    // Check if dimenions match. If dimensions are uninitialized assign them.
    if ((this->mWidth == 0) || (this->mHeight == 0) || (this->mChannels == 0)) {
        this->mWidth = src.mWidth;
        this->mHeight = src.mHeight;
        this->mChannels = src.mChannels;
        this->mStride = src.mStride;
    }
    else if (!compareDim(src, * this))
        throw cuCV::exception::DimensionMismatch(src, * this, "allocation");

    if (this->mData != NULL) {
        fprintf(stderr, "WARNING: Data on device is freed automatically when function allocateLike was called. Make sure to clear data yourself.");
        clearOnDevice();
    }

    /// Allocate Memory
    gpuErrchk(cudaMalloc((void**) & this->mData, sizeof(T) * src.mWidth * src.mHeight * src.mChannels));
    
    if (this->mData == NULL)
        throw std::bad_alloc(); //("Allocation of VRAM failed.")
}


template <typename T>
void cuCV::CuMat<T>::allocateOnDevice() {
    if (this->mData != NULL) {
        fprintf(stderr, "WARNING: Data on device is freed automatically when function allocateLike was called. Make sure to clear data yourself.");
        clearOnDevice();
    }
    if ((this->mWidth == 0) || (this->mHeight == 0) || (this->mChannels == 0))
        throw std::bad_alloc();  ///

    gpuErrchk(cudaMalloc((void**) & this->mData, sizeof(T) * this->mWidth * this->mHeight * this->mChannels));
    
    if (this->mData == NULL)
        throw std::bad_alloc(); //("Allocation of VRAM failed.")    
}


template <typename T>
void cuCV::CuMat<T>::clearOnDevice() {
    if (this->mData != NULL) {
        cudaFree(this->mData);
        this->mData = NULL;
    }
}


template <typename T>
cuCV::DeviceCuMat<T> cuCV::CuMat<T>::kernel() const {
    return cuCV::DeviceCuMat<T>(* this);
}


template <typename T>
bool cuCV::CuMat<T>::compareDim(const CuMat & A, const CuMat & B) const {
    if (A.mWidth != B.mWidth || A.mHeight != B.mHeight || A.mChannels != B.mChannels)
        return 0;
    return 1;
}

template <typename T>
bool cuCV::CuMat<T>::compareDim(const CuMat & A, const cuCV::Mat<T> & B) const {
    if (A.mWidth != B.mWidth || A.mHeight != B.mHeight || A.mChannels != B.mChannels)
        return 0;
    return 1;
}

template <typename T>
bool cuCV::CuMat<T>::compareDim(const cuCV::Mat<T> & A, const CuMat & B) const {
    if (A.mWidth != B.mWidth || A.mHeight != B.mHeight || A.mChannels != B.mChannels)
        return 0;
    return 1;
}

template <typename T>
bool cuCV::CuMat<T>::compareDim(const cuCV::Mat<T> & A, const cuCV::Mat<T> & B) const {
    if (A.mWidth != B.mWidth || A.mHeight != B.mHeight || A.mChannels != B.mChannels)
        return 0;
    return 1;
}


template <typename T>
bool cuCV::CuMat<T>::empty() const {
    if (this->mData == NULL)
        return true;
    else
        return false;
}


/// Explicit template specialization
template class cuCV::CuMat<CUCV_8U>;
template class cuCV::CuMat<CUCV_16U>;
template class cuCV::CuMat<CUCV_64F>;
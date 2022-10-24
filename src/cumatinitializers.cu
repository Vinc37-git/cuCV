/**
 * @file cumatinitializers.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "cumatinitializers.h"


template <typename T>
cuCV::CuMat<T> cuCV::getEmptyOnDevice(int width, int height, int channels) {
    cuCV::CuMat<T> mat(width, height, channels);
    //mat.allocateOnDevice();
    return mat;
}


template <typename T>
cuCV::CuMat<T> cuCV::zerosOnDevice(int width, int height, int channels) {
    cuCV::CuMat<T> mat(width, height, channels);
    mat.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((mat.getWidth() + threads.x - 1) / threads.x, (mat.getHeight() + threads.y - 1) / threads.y, mat.getNChannels());

    /// Perform Math
    cuCV::kernel::zeros<<<blocks, threads>>>(mat.kernel());

    #if CUCV_DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    return mat;
}


template <typename T>
cuCV::CuMat<T> cuCV::onesOnDevice(int width, int height, int channels) {
    cuCV::CuMat<T> mat(width, height, channels);
    mat.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((mat.getWidth() + threads.x - 1) / threads.x, (mat.getHeight() + threads.y - 1) / threads.y, mat.getNChannels());

    /// Perform Math
    cuCV::kernel::ones<<<blocks, threads>>>(mat.kernel());

    #if CUCV_DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    return mat;
}


template <typename T>
cuCV::CuMat<T> cuCV::eyeOnDevice(int width, int height, int channels) {
    cuCV::CuMat<T> mat(width, height, channels);
    mat.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((mat.getWidth() + threads.x - 1) / threads.x, (mat.getHeight() + threads.y - 1) / threads.y, mat.getNChannels());

    /// Perform Math
    cuCV::kernel::eye<<<blocks, threads>>>(mat.kernel());

    #if CUCV_DEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    #endif

    return mat;
}


template <typename T>
cuCV::CuMat<T> cuCV::gaussOnDevice(int width, int height, int channels, double sigma, bool norm) {
    if (width % 2 == 0 || height % 2 == 0) 
        throw std::runtime_error("Invalid side-length for Gaussian Kernel. It must be an odd number."); 
    if (!(sigma > 0))
        throw std::runtime_error("Invalid sigma for Gaussian Kernel. It must be > 0."); 

    cuCV::CuMat<T> mat(width, height, channels);
    mat.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((mat.getWidth() + threads.x - 1) / threads.x, (mat.getHeight() + threads.y - 1) / threads.y, mat.getNChannels());

    /// Perform Math
    float * pSum;
    gpuErrchk(cudaMalloc((void**) & pSum, sizeof(float)));  // void**) &
    gpuErrchk(cudaMemset((void *) pSum, 0, sizeof(float)));

    cuCV::kernel::gauss<<<blocks, threads>>>(mat.kernel(), sigma, norm, pSum);

    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());

    /// Since we can not gurantee synchronization on grid level of all threads in the gauss kernel after cummulating the sum,
    /// we will use a second, consecutive kernel to normalize the data
    if ((bool) norm) {
        //cuCV::kernel::gauss<<<blocks, threads>>>(mat.kernel(), sigma, norm, pSum);
        cuCV::kernel::div<<<blocks, threads>>>(mat.kernel(), mat.kernel(), pSum);
        #if CUCV_DEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        #endif
    }

    gpuErrchk(cudaFree(pSum));

    return mat;
}


/// Explicit template specialization
template cuCV::CuMat<CUCV_8U> cuCV::getEmptyOnDevice<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_16U> cuCV::getEmptyOnDevice<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_32F> cuCV::getEmptyOnDevice<CUCV_32F>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_64F> cuCV::getEmptyOnDevice<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::CuMat<CUCV_8U> cuCV::zerosOnDevice<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_16U> cuCV::zerosOnDevice<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_32F> cuCV::zerosOnDevice<CUCV_32F>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_64F> cuCV::zerosOnDevice<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::CuMat<CUCV_8U> cuCV::onesOnDevice<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_16U> cuCV::onesOnDevice<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_32F> cuCV::onesOnDevice<CUCV_32F>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_64F> cuCV::onesOnDevice<CUCV_64F>(const int width, const int height, const int channels);

template cuCV::CuMat<CUCV_8U> cuCV::eyeOnDevice<CUCV_8U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_16U> cuCV::eyeOnDevice<CUCV_16U>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_32F> cuCV::eyeOnDevice<CUCV_32F>(const int width, const int height, const int channels);
template cuCV::CuMat<CUCV_64F> cuCV::eyeOnDevice<CUCV_64F>(const int width, const int height, const int channels);

//template cuCV::CuMat<CUCV_8U> cuCV::gaussOnDevice(int length, int channels, double sigma, bool norm);
//template cuCV::CuMat<CUCV_16U> cuCV::gaussOnDevice(int length, int channels, double sigma, bool norm);
template cuCV::CuMat<CUCV_32F> cuCV::gaussOnDevice(int width, int height, int channels, double sigma, bool norm);
template cuCV::CuMat<CUCV_64F> cuCV::gaussOnDevice(int width, int height, int channels, double sigma, bool norm);
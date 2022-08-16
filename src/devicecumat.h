/**
 * @file kernelcumat.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef DEVICECUMAT_H
#define DEVICECUMAT_H

#include "cumat.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists

/**
 * @brief Kernel CuMat Class is a sibling of the standard CuMat Class. However, it only borrows a reference to the data
 * of an cuMat Object to launch the cuda kernel. This is neccessary due to the fact, that __global__ cuda kernels require 
 * arguments to be passed by value, which would result in a copy of data when using the standard cuMat object.
 * Since the data is borrowed only, the DeviceCuMat object will not take care of the data. When the object goes out of scope
 * after the kernel call, it will not free the data associated with the borrowed reference.
 * 
 * @tparam T 
 */
template <typename T>
class DeviceCuMat {
public:

    __host__ 
    DeviceCuMat(const CuMat<T> & cuMat);

    __device__ 
    DeviceCuMat(int width, int height, int channels, int stride);

    __device__ 
    cuCV::DeviceCuMat<T> getSubCuMat(int blockIdRow, int blockIdCol, int blockIdCh=0) const;


    __device__ 
    void setElement(const int row, const int col, const T value);


    __device__ 
    T getElement(const int row, const int col) const;


    __device__ 
    T getElement(const int row, const int col, const int ch) const;



    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mStride;  ///< Stride of the matrix represented by the mat object.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.
};
};

#endif  // DEVICECUMAT_H
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
class CuMat;  // Forward Declaration of CuMat to make sure compiler knows the class exists

/**
 * @brief Device CuMat Class is almost a copy of the standard CuMat Class. However, it only borrows a reference to the data
 * of an cuMat Object to launch the cuda kernel. This is neccessary due to the fact, that __global__ cuda kernels require 
 * arguments to be passed by value, which would result in a copy of data when using the standard cuMat object.
 * Since the data is borrowed only, the DeviceCuMat object will not take care of the data. When the object goes out of scope
 * after the kernel call, it will not free the data associated with the borrowed reference.
 * Note that most of the member functions are __device__ functions. Hence, it should not be used on the host.
 * Use it only for kernel calls else the standard CuMat Class
 * 
 * @tparam T 
 */
template <typename T>
class DeviceCuMat {
public:

    __host__ 
    DeviceCuMat(const CuMat<T> & cuMat);


    __device__ 
    DeviceCuMat(int width, int height, int channels, int strideX, int strideY);


    __device__ 
    cuCV::DeviceCuMat<T> getBlock(int blockIdRow, int blockIdCol, int blockIdCh=0) const;


    __device__ 
    cuCV::DeviceCuMat<T> getSubCuMat(const int row, const int col, const int ch, const int width, const int height) const;


    __device__ 
    void setElement(const int row, const int col, const T value);


    __device__ 
    void setElement(const int row, const int col, const int ch, const T value);


    __device__ 
    T getElement(const int row, const int col) const;


    __device__ 
    T getElement(const int row, const int col, const int ch) const;


///< @todo make private
    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mStrideX;  ///< Stride of the memory in x direction.
    int mStrideY;  ///< Stride of the memory in y direction.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.
};
};

#endif  // DEVICECUMAT_H
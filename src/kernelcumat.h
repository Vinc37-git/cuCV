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

#ifndef KERNELCUMAT_H
#define KERNELCUMAT_H

#include "cumat.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists

/**
 * @brief Kernel CuMat Class is a sibling of the standard CuMat Class. However, it only borrows a reference to the data
 * of an cuMat Object to launch the cuda kernel. This is neccessary due to the fact, that __global__ cuda kernels require 
 * arguments to be passed by value, which would result in a copy of data when using the standard cuMat object.
 * Since the data is borrowed only, the KernelCuMat object will not take care of the data. When the object goes out of scope
 * after the kernel call, it will not free the data associated with the borrowed reference.
 * 
 * @tparam T 
 */
template <typename T>
class KernelCuMat {
public:
    /**
     * @brief Construct a new Kernel Cu Mat object
     */
    //KernelCuMat();

    __host__ 
    KernelCuMat(const CuMat<T> & cuMat);

    __device__ 
    KernelCuMat(int width, int height, int channels, int stride);


    /**
     * @brief Construct a new Kernel Cu Mat object using a cuMat object.
     * 
     * @param cuMat 
     */
    //__host__ __device__ KernelCuMat(const CuMat<T> & cuMat);


    /**
     * @brief Construct a new Kernel Cu Mat object by copying another kernelCuMat object.
     * 
     * @param kernelCuMat 
     */
    //__host__ __device__ KernelCuMat(const KernelCuMat<T> & kernelCuMat);


    /**
     * @brief Destroy the Kernel Cu Mat object. No data will be freed.
     */
    //__host__ __device__ ~KernelCuMat();

    __device__ 
    cuCV::KernelCuMat<T> getSubCuMat(int blockIdRow, int blockIdCol, int blockIdCh=0) const;


    __device__ 
    void setElement(const int row, const int col, const T value);


    __device__ 
    T getElement(const int row, const int col) const;



    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mStride;  ///< Stride of the matrix represented by the mat object.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.
};
};

#endif  // KERNELCUMAT_H
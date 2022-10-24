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
 * @brief Device CuMat Class should be used for CUDA kernel calls.
 * Device CuMat Class is almost a copy of the standard CuMat Class. However, it only borrows a reference to the data
 * of an cuMat Object to launch the cuda kernel. This is neccessary due to the fact, that __global__ cuda kernels require 
 * arguments to be passed by value, which would result in a copy of data when using the standard cuMat object.
 * Since the data is borrowed only, the DeviceCuMat object will not take care of the data. When the object goes out of scope
 * after the kernel call, it will not free the data associated with the borrowed reference.
 * Note that most of the member functions are __device__ functions. Hence, it should not be used on the host.
 * Use it only for kernel calls else the standard CuMat Class. The standard CuMat objects can be returned as a Device CuMat obect
 * using the build in method `kernel()`. 
 * @todo: write docu for methods.
 * 
 * @tparam T The CUCV datatype.
 */
template <typename T>
class DeviceCuMat {
public:

    /**
     * @brief Construct a DeviceCuMat Objekt using a cuMat object. 
     * Note that it is not intended to use this constructor for a kernel call.
     * Rather use the kernel() method of CuMat Objects.
     * 
     * @param cuMat 
     */
    __host__ 
    DeviceCuMat(const CuMat<T> & cuMat);


    /**
     * @brief Construct a DeviceCuMat Objekt using the following parameter. 
     * Note that it is not intended to use this constructor for a kernel call.
     * Rather use the kernel() method of CuMat Objects.
     * 
     * @param width 
     * @param height 
     * @param channels 
     * @param strideX 
     * @param strideY 
     */
    __device__ 
    DeviceCuMat(int width, int height, int channels, int strideX, int strideY);


    /** @brief Get Sub Matrix of matrix, where width and height is the blockDim. 
     * Specify the blockId on the grid. */
    __device__ 
    cuCV::DeviceCuMat<T> getBlock(int blockIdRow, int blockIdCol, int blockIdCh=0) const;


    /** @brief Get Sub Matrix of matrix. Specify the first element (upper left corner) and width an height.*/
    __device__ 
    cuCV::DeviceCuMat<T> getSubCuMat(const int row, const int col, const int ch, const int width, const int height) const;


    /** @brief set element of matrix at specific 2D Position. */
    __device__ 
    void setElement(const int row, const int col, const T value);


    /** @brief Set element of matrix at specific 3D Position. */
    __device__ 
    void setElement(const int row, const int col, const int ch, const T value);


    /** @brief Get element of matrix at specific 2D Position. */
    __device__ 
    T getElement(const int row, const int col) const;


    /** @brief Get element of matrix at specific 3D Position. */
    __device__ 
    T getElement(const int row, const int col, const int ch) const;


    // ******* PUBLIC GETTERS *******
    
    /** @brief Get the width of Mat. */
    __device__ 
    int getWidth() const;

    /** @brief Get the height of Mat. */
    __device__ 
    int getHeight() const;

    /** @brief Get the depth of Mat. */
    __device__ 
    int getNChannels() const;

    /** @brief Get the Stride in width-direction of Mat. */
    __device__ 
    int getStrideX() const;

    /** @brief Get the Stride in height-direction of Mat. */
    __device__ 
    int getStrideY() const;

    /** @brief Get the Data pointer pointing to the first element of Mat. */
    __device__ 
    T * getDataPtr() const;

    /** @brief Get thenumber of elements of Mat. */
    __device__ 
    size_t getSize() const;


private:
    int mWidth;  ///< Width of the matrix represented by the mat object.
    int mHeight;  ///< Height of the matrix represented by the mat object.
    int mStrideX;  ///< Stride of the memory in x direction.
    int mStrideY;  ///< Stride of the memory in y direction.
    int mChannels;  ///< Number of channels of the matrix represented by the mat object.
    T * mData;  ///< Pointer to the data of the matrix represented by the mat object.
};
};

#endif  // DEVICECUMAT_H
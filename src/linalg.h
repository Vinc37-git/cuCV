/**
 * @file linalg.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef LINALG_H
#define LINALG_H

#include <iostream>
#include <unistd.h>

#include "errorhandling.h"
#include "mat.h"
#include "kernel.h"
#include "devicecumat.h"
#include "filter.h"


namespace cuCV {


enum class Kernel; // Forward Delcaration of enum class Kernel to make sure compiler knows it.


/** @brief Add one CuMat object A to the other B elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T>
void add(CuMat<T> & OUT, const CuMat<T> & A, const CuMat<T>& B);


/** @brief Add scalar alpha to a CuMat object B elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T>
void add(CuMat<T> & OUT, const CuMat<T> & A, const T alpha);


/** @brief Subtract one CuMat object B from the other A elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T>
void dif(CuMat<T>& OUT, const CuMat<T>& A, const CuMat<T>& B);


/** @brief Subtract a scalar alpha from the other A elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T>
void dif(CuMat<T> & OUT, const CuMat<T> & A, const T alpha);


/** @brief Multiply one CuMat object A with the other B elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically. 
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T>
void mul(CuMat<T> & OUT, const CuMat<T> & A, const CuMat<T>& B);


/** @brief Multiply one CuMat object A with a scalar alpha elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically. 
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T>
void mul(CuMat<T> & OUT, const CuMat<T> & A, const T alpha);


/** @brief Divide one CuMat object A by the other B elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T>
void div(CuMat<T> & OUT, const CuMat<T> & A, const CuMat<T>& B);


/** @brief Divide one CuMat object A by a scalar alpha elementwise and set results in object OUT. 
 * OUT can be prealloceted to speed up the computation. Otherwise the memory will be allocated automatically.
 * @param OUT The Matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T>
void div(CuMat<T> & OUT, const CuMat<T> & A, const T alpha);


/**
 * @brief Calculate the matrix multiplication of two matrices on the device 
 * without using the concept of shared memory. 
 * Pass the matrix which will keep the result by reference.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_64F
 * @param OUT The matrix which will hold the resulting matrix of shape NxL.
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 */
template <typename T>
void simpleMatmul(CuMat<T> & OUT, const CuMat<T> & A, const CuMat<T> & B);


/**
 * @brief Calculate the matrix multiplication of two matrices on the device 
 * without using the concept of shared memory and return the result.
 * @todo rename to simpleMatmul()
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_64F
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 * @return The resulting matrix with shape NxL.
 */
template <typename T>
CuMat<T> naiveMatmul(const CuMat<T> & A, const CuMat<T> & B);


/**
 * @brief Calculate the matrix multiplication of two matrices on the device 
 * using the concept of shared memory. 
 * Pass the matrix which will keep the result by reference.
 * @todo some words about shared memory.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_64F
 * @param OUT The matrix which will hold the resulting matrix of shape NxL.
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 */
template <typename T>
void matmul(CuMat<T> & OUT, const CuMat<T> & A, const CuMat<T> & B);


/**
 * @brief Calculate the matrix multiplication of two matrices on the device 
 * using the concept of shared memory and return the result.
 * @todo some words about shared memory.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_64F
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 * @return The resulting matrix with shape NxL of type T.
 */
template <typename T>
CuMat<T> matmul(const CuMat<T> & A, const CuMat<T> & B);


/**
 * @note REMOVE THIS FUNCTION and replace with boxFilter, gaussFilter, etc...
 */
template <typename T>
CuMat<T> simpleConv2d(const CuMat<T> & A, const cuCV::Kernel kernel, const size_t kernelX, const size_t kernelY=3, const cuCV::Padding padding=0);


/**
 * @brief Perform a convolution on `A` by passing a kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * Pass the matrix which will keep the result by reference.
 * @todo rename to simpleConv2d
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_64F.
 * @param OUT The matrix which will keep the result.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 * @return the resulting matrix of type T1. The size dependes on the Padding method.
 */
template <typename T1, typename T2>
void simpleConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


/**
 * @brief Perform a convolution on `A` by passing a kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * @todo rename to simpleConv2d
 * 
 * @tparam T1 CUCV datatype of return and `A`: CUCV_8U, CUCV_16U, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_64F.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 * @return the resulting matrix of type T1. The size dependes on the Padding method.
 */
template <typename T1, typename T2>
CuMat<T1> simpleConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


/**
 * @brief Perform a convolution on `A` by passing a kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * Pass the matrix which will keep the result by reference.
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_64F.
 * @param OUT The matrix which will keep the result.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 * @return the resulting matrix of type T1. The size dependes on the Padding method.
 */
template <typename T1, typename T2>
void simpleSharedConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


/**
 * @brief Perform a convolution on `A` by passing a kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * 
 * @tparam T1 CUCV datatype of return and `A`: CUCV_8U, CUCV_16U, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_64F.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 * @return the resulting matrix of type T1. The size dependes on the Padding method.
 */
template <typename T1, typename T2>
CuMat<T1> simpleSharedConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


template <typename T1, typename T2>
void sharedPaddingConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);

template <typename T1, typename T2>
CuMat<T1> sharedPaddingConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


template <typename T1, typename T2>
void sepSharedConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & rowKernel, const CuMat<T2> & colKernel, const cuCV::Padding padding);

template <typename T1, typename T2>
CuMat<T1> sepSharedConv2d(const CuMat<T1> & A, const CuMat<T2> & rowKernel, const CuMat<T2> & colKernel, const cuCV::Padding padding);


namespace convHelper {


template <typename T>
void estimateOutSize(const CuMat<T> & A, const Padding padding , int * width, int * height);


template <typename T1, typename T2>
void checks(const CuMat<T1> & A, const CuMat<T2> & kernel);


/**
 * @brief This function takes the number of bytes of an array and checks 
 * if it is a multiple of sizeofNextArrayType. If not, the function returns 
 * a recommended length of array, to ensure alignment of the next array. 
 * @param arrayCounts number of bytes of the array.
 * @param sizeofNextArrayType size of the next array type.
 */
size_t cucvAddBytes2Align(size_t arrayCounts, size_t sizeofNextArrayType); 


}  // namespace convHelper
}  // namespace cuCV

#endif  // LINALG_H
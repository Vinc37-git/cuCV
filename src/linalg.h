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


/**
 * @brief Calculate the matrix multiplication of two matrices on the device 
 * without using the concept of shared memory.
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
CuMat<T> slowConv2d(const CuMat<T> & A, const cuCV::Kernel kernel, const size_t kernelX, const size_t kernelY=3, const cuCV::Padding padding=0);


/**
 * @brief Perform a convolution on `A` by passing a kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * @todo rename to simpleConv2d
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_64F.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 * @return the resulting matrix of type T1. The size dependes on the Padding method.
 */
template <typename T1, typename T2>
CuMat<T1> slowConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


}  // namespace cuCV

#endif  // LINALG_H
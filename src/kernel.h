/**
 * @file kernel.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#include "mat.h"
#include "cumat.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists


template <typename T>
class KernelCuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists


namespace kernel {


template <typename T>
__global__ void add(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

template <typename T>
__global__ void add(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const T alpha);


template <typename T>
__global__ void dif(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

template <typename T>
__global__ void dif(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const T alpha);


template <typename T>
__global__ void mul(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

template <typename T>
__global__ void mul(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const T alpha);


template <typename T>
__global__ void div(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

template <typename T>
__global__ void div(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const T alpha);

template <typename T>
__global__ void matmul(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

template <typename T>
__global__ void naiveMatmul(cuCV::KernelCuMat<T> OUT, const cuCV::KernelCuMat<T> A, const cuCV::KernelCuMat<T> B);

};  // namespace kernel
};  // namespace cuCV


#endif //
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
#include "filter.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists


template <typename T>
class DeviceCuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists

enum class Padding;


namespace kernel {


template <typename T> __global__ 
void add(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T> __global__ 
void add(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


template <typename T> __global__ 
void dif(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T> __global__ 
void dif(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


template <typename T> __global__ 
void mul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T> __global__ 
void mul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


template <typename T> __global__ 
void div(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T> __global__ 
void div(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


template <typename T> __global__ 
void naiveMatmul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T> __global__ 
void matmul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


template <typename T1, typename T2> __global__
void slowConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding);


template <typename T> __global__ 
void zeros(cuCV::DeviceCuMat<T> OUT);


template <typename T> __global__ 
void ones(cuCV::DeviceCuMat<T> OUT);


template <typename T> __global__ 
void eye(cuCV::DeviceCuMat<T> OUT);


};  // namespace kernel
};  // namespace cuCV


#endif //
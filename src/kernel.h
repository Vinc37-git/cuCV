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

namespace kernel {

template< typename T >
__global__ void add(cuCV::CuMat<T> OUT, const cuCV::CuMat<T> A, const cuCV::CuMat<T> B);


template< typename T >
__global__ void dif(cuCV::CuMat<T> OUT, cuCV::CuMat<T> A, cuCV::CuMat<T> B);


template< typename T >
__global__ void mul(cuCV::CuMat<T> OUT, cuCV::CuMat<T> A, cuCV::CuMat<T> B);


template< typename T >
__global__ void div(cuCV::CuMat<T> OUT, cuCV::CuMat<T> A, cuCV::CuMat<T> B);

};  // namespace kernel
};  // namespace cuCV


#endif //
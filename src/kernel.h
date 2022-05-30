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
class CuMat;  ///< Forward Declaration to make sure compiler knows the class exists

namespace kernel {

//template< typename T >
//__global__ void add(T * OUT);

template< typename T >
__global__ void dif(CuMat<T> OUT);

template< typename T >
__global__ void add(cuCV::CuMat<T> & OUT, cuCV::CuMat<T> & A, cuCV::CuMat<T> & B);
//__global__ void dif(cuCV::CuMat & OUT, cuCV::CuMat & A, cuCV::CuMat & B);
//__global__ void mul(cuCV::CuMat & OUT, cuCV::CuMat & A, cuCV::CuMat & B);
//__global__ void div(cuCV::CuMat & OUT, cuCV::CuMat & A, cuCV::CuMat & B);
};
};


#endif //
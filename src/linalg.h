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


template <typename T>
CuMat<T> naiveMatmul(const CuMat<T> & A, const CuMat<T> & B);


template <typename T>
CuMat<T> matmul(const CuMat<T> & A, const CuMat<T> & B);


template <typename T>
CuMat<T> slowConv2d(const CuMat<T> & A, const cuCV::Kernel, const size_t kernelX, const size_t kernelY, const cuCV::Padding);


template <typename T1, typename T2>
CuMat<T1> slowConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding);


}  // namespace cuCV

#endif  // LINALG_H
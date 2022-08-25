/**
 * @file filter.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef FILTER_H
#define FILTER_H

#include "cumat.h"
#include "cumatinitializers.h"

template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists

namespace cuCV {

enum class Padding {/*NONE,*/ ZERO/*, SAME*/};

enum class Kernel {BOX, BOX_UNNORM, SOBELX, SOBELY, LAPLACE, GAUSS};

template <typename T>
CuMat<T> createKernel(const Kernel kerneltype, const size_t kernelX, const size_t kernelY);
    
}  // namespace cuCV

#endif  // FILTER_H
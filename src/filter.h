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
class CuMat;  // Forward Declaration of CuMat to make sure compiler knows the class exists

namespace cuCV {

enum class Padding {/*NONE,*/ ZERO/*, SAME*/};


/**
 * @brief Kernels which are directly available in cuCV. 
 * 
 */
enum class Kernel {
    BOX,  ///< Box Filter, also known as moving average.
    BOX_UNNORM,  ///< Unnormalized Box filter. The sum over weights equals the kernel size.
    SOBELX,  ///< Edge detection in X direction.
    SOBELY,  ///< Edge detection in Y direction.
    LAPLACE,  ///< Discrete approximation of Laplace operator in both x and y direction (2nd derivation).
    GAUSS  ///< Filter which entries are gaussian distributed.
};


/**
 * @brief Create a Kernel (or filter). You can choose between different predefined kernel types and size.
 * If you create a Gaussian distributed kernel, choose a standart deviation.
 * Note that SobelX, SobelY and Laplace have predefined size (3x3) and Gaussian kernel must be square.
 * 
 * @tparam T CuType of the Kernel. Note that most of the kernels should be of floating type as comma values will be truncated otherwise.
 * @param kerneltype Choose between Box, unnormalized Box, SobelX, SobelY, Laplace and Gauss.
 * @param kernelX Kernel size in X direction for Box and Gaussian kernels. Defaults to 3.
 * @param kernelY Kernel size in Y direction for Box and Gaussian kernels. Defaults to 3.
 * @param sigma The standard deviation for Gaussian kernels. Defaults to 1.
 * @return The kernel of type T.
 */
template <typename T>
CuMat<T> createKernel(const Kernel kerneltype, const size_t kernelX=3, const size_t kernelY=3, const int sigma=1);
    
}  // namespace cuCV

#endif  // FILTER_H
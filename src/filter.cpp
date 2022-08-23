/**
 * @file filter.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "filter.h"

template <typename T>
cuCV::CuMat<T> createKernel(const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY) {

    switch (kerneltype){
        case cuCV::Kernel::BOX:
            return cuCV::onesOnDevice<CUCV_64F>(kernelX, kernelY, 1) / (kernelX * kernelY);;
        
        case cuCV::Kernel::BOX_UNNORM:
            return cuCV::onesOnDevice<CUCV_64F>(kernelX, kernelY, 1);

        case cuCV::Kernel::SOBELX:
            CUCV_64F sobelXraw[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
            cuCV::Mat<CUCV_64F> sobelXcpu(3, 3, 1, sobelXraw);
            cuCV::CuMat<CUCV_64F> sobelXdev(3, 3, 1);
            sobelXdev.uploadFrom(sobelXcpu);
            return sobelXdev;
        
        case cuCV::Kernel::SOBELY:
            CUCV_64F sobelYraw[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
            cuCV::Mat<CUCV_64F> sobelYcpu(3, 3, 1, sobelYraw);
            cuCV::CuMat<CUCV_64F> sobelYdev(3, 3, 1);
            sobelYdev.uploadFrom(sobelYcpu);
            return sobelYdev;
        
        case cuCV::Kernel::LAPLACE:
            CUCV_64F laplaceRaw[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
            cuCV::Mat<CUCV_64F> laplaceCpu(3, 3, 1, laplaceRaw);
            cuCV::CuMat<CUCV_64F> laplaceDev(3, 3, 1);
            laplaceDev.uploadFrom(laplaceCpu);
            return laplaceDev;
        
        default:
            break;
    }
}
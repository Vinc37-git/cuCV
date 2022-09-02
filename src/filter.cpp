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

//template <typename T>
cuCV::CuMat<CUCV_64F> cuCV::createKernel(const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY, const int sigma) {

    if (kerneltype == cuCV::Kernel::BOX) {
            return cuCV::onesOnDevice<CUCV_64F>(kernelX, kernelY, 1) / (kernelX * kernelY);;
    }
    else if (kerneltype == cuCV::Kernel::BOX_UNNORM) {
            return cuCV::onesOnDevice<CUCV_64F>(kernelX, kernelY, 1);
    }
    else if (kerneltype == cuCV::Kernel::SOBELX) {
            CUCV_64F * sobelXraw = new CUCV_64F [9] {1, 0, -1, 2, 0, -2, 1, 0, -1};
            cuCV::Mat<CUCV_64F> sobelXcpu(3, 3, 1, sobelXraw, false);
            cuCV::CuMat<CUCV_64F> sobelXdev(3, 3, 1);
            sobelXdev.uploadFrom(sobelXcpu);
            return sobelXdev;
    }
    else if (kerneltype == cuCV::Kernel::SOBELY) {
            CUCV_64F * sobelYraw = new CUCV_64F [9] {1, 2, 1, 0, 0, 0, -1, -2, -1};
            cuCV::Mat<CUCV_64F> sobelYcpu(3, 3, 1, sobelYraw, false);
            cuCV::CuMat<CUCV_64F> sobelYdev(3, 3, 1);
            sobelYdev.uploadFrom(sobelYcpu);
            return sobelYdev;
    }
    else if (kerneltype == cuCV::Kernel::LAPLACE) {
            CUCV_64F * laplaceRaw = new CUCV_64F [9] {0, 1, 0, 1, -4, 1, 0, 1, 0};
            cuCV::Mat<CUCV_64F> laplaceCpu(3, 3, 1, laplaceRaw, false);
            cuCV::CuMat<CUCV_64F> laplaceDev(3, 3, 1);
            laplaceDev.uploadFrom(laplaceCpu);
            return laplaceDev;
    }
    else if (kerneltype == cuCV::Kernel::GAUSS) {
            bool norm = true;  ///< @todo: Pass as argument
            return cuCV::gaussOnDevice<CUCV_64F>(kernelX, 1, sigma, norm);
    }   
    else {
        fprintf(stderr, "Error: Unknown kernel type: %d", (int) kerneltype);
        exit(EXIT_FAILURE);
    }
}


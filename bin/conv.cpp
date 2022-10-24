#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cumat.h"
#include "linalg.h"


/**
 * @brief Use the profiler nvprof and profile Convolution.
 */
int main(int argc, char ** argv) {
    int N = 1024 * 2; // * 2 * 2;  // Max length of one matrix side.
    int k = 5;

    cuCV::CuMat<CUCV_32F> A = cuCV::onesOnDevice<CUCV_32F>(N, N, 1);
    cuCV::CuMat<CUCV_32F> K = cuCV::gaussOnDevice<CUCV_32F>(k, k, 1, 1);
    cuCV::CuMat<CUCV_32F> Kr = cuCV::gaussOnDevice<CUCV_32F>(k, 1, 1, 1);
    cuCV::CuMat<CUCV_32F> Kc = cuCV::gaussOnDevice<CUCV_32F>(1, k, 1, 1);
    cuCV::CuMat<CUCV_32F> C = cuCV::CuMat<CUCV_32F>(N, N, 1);

    cuCV::simpleConv2d(C, A, K, cuCV::Padding::ZERO);
    cuCV::simpleSharedConv2d(C, A, K, cuCV::Padding::ZERO);
    cuCV::sharedPaddingConv2d(C, A, K, cuCV::Padding::ZERO);
    cuCV::sepSharedConv2d(C, A, Kr, Kc, cuCV::Padding::ZERO);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include "cumat.h"
#include "kernel.h"


/**
 * @brief Use the profiler nvprof and profile Matrix Multiplication.
 */
int main(int argc, char ** argv) {
    int N = 1024 * 2 * 2; // * 2 * 2;  // Max length of one matrix side.

    cuCV::CuMat<CUCV_32F> A = cuCV::onesOnDevice<CUCV_32F>(N, N, 1);
    cuCV::CuMat<CUCV_32F> B = cuCV::onesOnDevice<CUCV_32F>(N, N, 1);
    cuCV::CuMat<CUCV_32F> C = cuCV::CuMat<CUCV_32F>(N, N, 1);

    cuCV::matmul(C, A, B);
    cuCV::simpleMatmul(C, A, B);

    return 0;
}
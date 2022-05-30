/**
 * @file main.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda_runtime.h>

#include "mat.h"
#include "cumat.h"



int main() {

    int N = 10000;

    cuCV::Mat A = cuCV::Mat<CUCV_8U>(N, N, 1);
    cuCV::Mat B = cuCV::Mat<CUCV_8U>(N, N, 1);
    cuCV::Mat C = cuCV::Mat<CUCV_8U>(N, N, 1);

    A.ones();
    B.ones();

    A = A + 5;

    A.print(5, 5);

    cuCV::CuMat<CUCV_8U> A_dev;
    cuCV::CuMat<CUCV_8U> B_dev;
    cuCV::CuMat<CUCV_8U> C_dev;

    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);

    A_dev.add(C_dev, B_dev);
    //C_dev = A_dev + B_dev;

    C_dev.downloadTo(C);




    //cuCV::CuMat B_dev = cuCV::CuMat(N, N, 1, b);
    //cuCV::CuMat C_dev = cuCV::CuMat(N, N, 1, c);

    //cudaMalloc(&A_dev.elements, sizeof(char) * N * N);
    //cudaMalloc(&B_dev.elements, sizeof(char) * N * N);
    //cudaMalloc(&C_dev.elements, sizeof(char) * N * N);


    /*

    // Transfer input data from host (CPU) to device (GPU)
    cudaMemcpy(A_dev.elements, a, sizeof(char) * N * N, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_dev.elements, b, sizeof(char) * N * N, cudaMemcpyHostToDevice); 

    // perform addition on GPU
    A_dev.add(B_dev, C_dev, N * N);

    // Free Memory of a_d und a_b
    cudaFree(A_dev.elements);
    cudaFree(B_dev.elements);

    // Transfer output data from device (GPU) to host (CPU)
    cudaMemcpy(c, C_dev.elements, sizeof(char) * N * N, cudaMemcpyDeviceToHost);

    // Free Memory of out_d
    cudaFree(C_dev.elements);

    // Verification
    //for(int i = 0; i < N; i++){
    //    assert(fabs(out[i] - a[i] - b[i]) < ERR_MAX);
    //}

    printf("out[0] = %i\n", c[0]);
    printf("PASSED\n");
    
    free(a);
    free(b);
    free(c);

    cudaDeviceSynchronize();
    */


    return EXIT_SUCCESS;
}
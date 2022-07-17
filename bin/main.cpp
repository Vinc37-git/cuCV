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
#include <unistd.h>

#include <cuda_runtime.h>

#include "mat.h"
#include "cumat.h"



int main() {
    fprintf(stderr, "Starting main.\n");

    int N = 4;

    cuCV::Mat A = cuCV::Mat<CUCV_64F>(N, N, N);
    cuCV::Mat B = cuCV::Mat<CUCV_64F>(N, N, N);
    cuCV::Mat C = cuCV::Mat<CUCV_8U>(N, N, N);

    /*A.ones();
    //B.ones();

    A += 5;

    A.print(5, 5, 0);

    B = A + 2;

    B.print(5, 5, 0);


    B = B + 2;

    B.print(5, 5, 1);*/

    C.eye();

    CUCV_64F testData[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    std::cout << "Ch1 " << std::endl;
    C.print(5,5,0);
    std::cout << "Ch2 " << std::endl;
    C.print(5,5,1);

    for (size_t i=0; i<8; i++) {
        std::cout << (int) C.mData[i] << " vs " << testData[i] <<std::endl;
        if (C.mData[i] != testData[i])
            printf("failed.\n");
    } 



    //A.clear();
    //B.clear();

return 0;

/*
    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_8U> B_dev(B);
    cuCV::CuMat<CUCV_8U> C_dev(C);

    try {
        std::cout << "Uploading A" << std::endl;
        A_dev.uploadFrom(A);

        std::cout << "Uploading B" << std::endl;
        B_dev.uploadFrom(B);

        std::cout << "Performing Math" << std::endl;
        A_dev.add(C_dev, B_dev);

        //C_dev = A_dev + B_dev;

        std::cout << "Downloading Results" << std::endl;
        C_dev.downloadTo(C);

        std::cout << "Print Solution: " << std::endl;
        C.print(5,5,0);
    }
    catch(const char * e) {
        std::cerr << "Catched: " << e << '\n';
    }
    catch(std::exception const & e) {
        std::cerr << e.what() << '\n';
    }
    

*/


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
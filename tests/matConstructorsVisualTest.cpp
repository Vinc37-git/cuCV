/**
 * @file matConstructorsVisualTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include "mat.h"

int main() {

    int N = 5;

    // Default Constructor
    cuCV::Mat stdMat = cuCV::Mat<CUCV_8U>();
    stdMat.~Mat();

    // Construct empty Mat that can be initialized with data
    cuCV::Mat emptyMat = cuCV::Mat<CUCV_8U>(N, N, N);
    std::cout << "Empty mat initialized with ones:" << std::endl;
    emptyMat.ones();
    emptyMat.print(5,5,0);
    std::cout << "\nEmpty mat initialized with eye:" << std::endl;
    emptyMat.eye();
    std::cout << "First Channel:" << std::endl;
    emptyMat.print(5,5,0);
    std::cout << "Second Channel:" << std::endl;
    emptyMat.print(5,5,1);
    emptyMat.~Mat();


    // Construct a Mat object and fill it with data already stored in memory.
    CUCV_64F * data = (CUCV_64F *) malloc(N * N * N * sizeof(CUCV_64F));
    for (int i=0; i<N*N*N; ++i) 
        data[i] = i;

    cuCV::Mat preAllocMat = cuCV::Mat<CUCV_64F>(N, N, N, data);

    std::cout << "\nPreallocated mat:" << std::endl;
    std::cout << "First Channel:" << std::endl;
    preAllocMat.print(5,5,0);
    std::cout << "Second Channel:" << std::endl;
    preAllocMat.print(5,5,1);
    preAllocMat.~Mat();

    return EXIT_SUCCESS;
}
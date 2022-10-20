/**
 * 
 * 
 * 
 * 
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include <cuda_runtime.h>

#include "errorhandling.h"

int main(int argc, char ** argv) {
    
    float * data, * devData;
    data = (float *) malloc(1000 * sizeof(float));

    for (int i=0; i<1000; ++i)
        data[i] = 5;

    gpuErrchk(cudaMalloc((void**) & devData, sizeof(float) * 1000));
    gpuErrchk(cudaMemcpy(devData ,data , sizeof(float) * 1000, cudaMemcpyHostToDevice));
    cudaFree(devData);
    free(data);

    gpuErrchk(cudaDeviceReset());
    return 0;
}
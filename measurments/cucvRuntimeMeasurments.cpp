/**
 * @file cucvconv.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <filesystem>


#include <typeinfo>

#ifndef CUCV_XX
#define CUCV_XX CUCV_8U  // otherwise CUCV_XX can be defined using cmakes target_compile_definitions()
#endif

#include "measurmentsHelper.cpp"
#include "addMeasurments.cpp"
#include "matmulMeasurments.cpp"
#include "conv2dMeasurments.cpp"

namespace fs = std::filesystem;

/**
 * @brief 
 * Log header:
 *  * Device:
 *      * 0 : Host
 *      * 1 : CUDAwDT
 *      * 2 : CUDAwoDT
 *      * 3 : CUDAwDTshared
 *      * 4 : CUDAwoDTshared
 *      * 99 : openCV
 *  * Run
 *  * Pixel: N * N * nCh
 *  * Time in ns
 */
int main(int argc, char ** argv) {
    int N = 1024 * 2 * 2 * 2 * 2;  // Max length of one matrix side.
    int nCh = 1;  // Number of channels
    int kSize = 3 * 3 * 3 * 3; // Max length of the kernel for convolution
    int I_MAX = 5;  // Perform every computation I_MAX times

    size_t countPerElem = sizeof(CUCV_XX);
    if (countPerElem > 4)
        N = std::min(N, 8192);  // otherwise matrices will require to much Memory

    // Note: WARMUP: Perform a warm-up computation outside of the timed computation to remove 
    // the CUDA startup overhead from performance measurements.
    //if (false)
    {
        // Measure Addition
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("addition_" + std::to_string(countPerElem) + "B.csv");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device;Run;N;Time\n");
        fprintf(log, "{0:'Host',1:'CUDAwDT',2:'CUDAwoDT',3:'CUDAwoDTprealloc',99:'openCV'};[i];[N * N * %i];[ns]\n", nCh);

        additionHost(N, nCh, I_MAX, log);
        additionCUDAwDT(N, nCh, I_MAX, log);
        additionCUDAwoDT(N, nCh, I_MAX, log);
        //additionCUDAwoDTprealloc(N, nCh, I_MAX, log);
        additionOpenCV(N, nCh, I_MAX, log);

        fclose(log);
    }
    //if (false)
    {
        // Measure Matrix Multiplication
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("matmul_" + std::to_string(countPerElem) + "B.csv");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device;Run;N;Time\n");
        fprintf(log, "{0:'Host',1:'CUDAwDT',2:'CUDAwoDT',3:'CUDAwDTshared',4:'CUDAwoDTshared',99:'openCV'};[i];I_dim[N * N * %i];[ns]\n", nCh);

        matmulCUDAwDT(N, nCh, I_MAX, log);
        matmulCUDAwoDT(N, nCh, I_MAX, log);
        matmulCUDAwDTshared(N, nCh, I_MAX, log);
        matmulCUDAwoDTshared(N, nCh, I_MAX, log);
        if (countPerElem >= 4)
            matmulOpenCV(N, nCh, I_MAX, log);  // matmul in openCV is only supported for floating point matrices

        fclose(log);
    }
    //if (false)
    {
        N = 2048;
        // Measure Convolution
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("convolution_" + std::to_string(countPerElem) + "B.csv");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device;Run;N;M;Time\n");
        fprintf(log, "{0:'Host',1:'simpleConv2DwDT',2:'simpleConv2DwoDT',3:'simpleSharedConv2DwDT',4:'simpleSharedConv2DwoDT',"
        "5:'sharedPaddingConv2DwDT',6:'sharedPaddingConv2DwoDT',7:'sepSharedConv2DwDT',8:'sepSharedConv2DwoDT',99:'openCV'};"
        "[i];I_dim [N * N * %i];k_dim [M * M];[ns]\n", nCh);

        conv2dCUDAwDT(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwoDT(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwDTshared(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwoDTshared(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwDTsharedPadded(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwoDTsharedPadded(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwDTsharedSeparated(N, nCh, kSize, I_MAX, log);
        conv2dCUDAwoDTsharedSeparated(N, nCh, kSize, I_MAX, log);
        conv2dOpenCV(N, nCh, kSize, I_MAX, log);

        fclose(log);
    }
}

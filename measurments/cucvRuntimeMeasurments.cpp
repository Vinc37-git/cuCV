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

#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "linalg.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <typeinfo>

#ifndef CUCV_XX
#define CUCV_XX CUCV_8U  // otherwise CUCV_XX can be defined using cmakes target_compile_definitions()
#endif

namespace fs = std::filesystem;

std::chrono::steady_clock::time_point begin() {
    return std::chrono::steady_clock::now();
}


int64_t stop(std::chrono::steady_clock::time_point begin) {
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();
} 


int getCVequivalent(int nCh) {
    int openCVdtype = -1;

    if (typeid(CUCV_XX) == typeid(CUCV_8U) && nCh == 3) 
        openCVdtype = CV_8UC3;
    else if (typeid(CUCV_XX) == typeid(CUCV_16U) && nCh == 3) 
        openCVdtype = CV_16UC3;
    else if (typeid(CUCV_XX) == typeid(CUCV_64F) && nCh == 3) 
        openCVdtype = CV_64FC3;
    else if (typeid(CUCV_XX) == typeid(CUCV_8U) && nCh == 1) 
        openCVdtype = CV_8U;
    
    else if (typeid(CUCV_XX) == typeid(CUCV_16U) && nCh == 1) 
        openCVdtype = CV_16U;
    else if (typeid(CUCV_XX) == typeid(CUCV_64F) && nCh == 1) 
        openCVdtype = CV_64F;
    else {
        fprintf(stderr, "No openCV datatype equivalent found.");
        abort();
    }

    return openCVdtype;
}


/**
 * @brief Runtime Measurement of Matrix Addition on host (CPU).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void additionHost(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix addition with size N * N * %i on HOST\n", nCh); 
    int device = 0;  // host

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);

            auto tic = begin();
            
            cuCV::Mat cucvMat3 = cucvMat1 + cucvMat2;

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Addition on CUDA device considering upload and download time.
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void additionCUDAwDT(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix addition with size N * N * %i on CUDA device with up-/download.\n", nCh); 
    int device = 1;  // CUDAwDT

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            cuCV::CuMat cucvMat3_dev = cucvMat1_dev + cucvMat2_dev;
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Addition on CUDA device without considering upload and download time.
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void additionCUDAwoDT(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix addition with size N * N * %i on CUDA device without up-/download.\n", nCh); 
    int device = 2;  // CUDAwoDT

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            
            auto tic = begin();
            cuCV::CuMat cucvMat3_dev = cucvMat1_dev + cucvMat2_dev;
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Addition on host using openCV.
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void additionOpenCV(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix addition with size N * N * %i on host using openCV.\n", nCh); 
    int device = 5;  // openCV

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cv::Mat opencvMat1 = cv::Mat::ones(n, n, getCVequivalent(nCh));
            cv::Mat opencvMat2 = cv::Mat::ones(n, n, getCVequivalent(nCh));

            cv::randu(opencvMat1, cv::Scalar(0), cv::Scalar(256));
            
            auto tic = begin();
            cv::Mat opencvMat3 = opencvMat1 + opencvMat2;

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


void matmulHost(int N, int nCh, int i_max, FILE * file) {
    /* not available. Maybe I ll add it. */
}


/**
 * @brief Runtime Measurement of Matrix Multiplication on CUDA device considering upload and download time (data transfer(DT)).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void matmulCUDAwDT(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix multiplication with size N * N * %i on CUDA device with up-/download.\n", nCh); 
    int device = 1;  // CUDAwDT

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            cuCV::CuMat cucvMat3_dev = cuCV::matmul(cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Multiplication on CUDA device without considering upload and download time (data transfer).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void matmulCUDAwoDT(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix multiplication with size N * N * %i on CUDA device without up-/download.\n", nCh); 
    int device = 2;  // CUDAwoDT

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            
            auto tic = begin();
            cuCV::CuMat cucvMat3_dev = cuCV::matmul(cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Multiplication on CUDA device using shared memory and considering upload and download time (data transfer(DT)).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void matmulCUDAwDTshared(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix multiplication with size N * N * %i on CUDA device using shared memory with up-/download.\n", nCh); 
    int device = 3;  // CUDAwDTshared

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            cuCV::CuMat cucvMat3_dev = cuCV::naiveMatmul(cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Multiplication on CUDA device using shared memory and without considering upload and download time (data transfer).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void matmulCUDAwoDTshared(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix multiplication with size N * N * %i on CUDA device using shared memory without up-/download.\n", nCh); 
    int device = 4;  // CUDAwoDTshared

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            
            auto tic = begin();
            cuCV::CuMat cucvMat3_dev = cuCV::naiveMatmul(cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of Matrix Multiplication on host using openCV
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void matmulOpenCV(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of matrix multiplication with size N * N * %i on host using openCV.\n", nCh); 
    int device = 5;  // openCV

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cv::Mat opencvMat1 = cv::Mat::ones(n, n, getCVequivalent(nCh));
            cv::Mat opencvMat2 = cv::Mat::ones(n, n, getCVequivalent(nCh));

            cv::randu(opencvMat1, cv::Scalar(0), cv::Scalar(256));
            
            auto tic = begin();
            cv::Mat opencvMat3 = opencvMat1 * opencvMat2;

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device considering upload and download time (data transfer(DT)).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwDT(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device with up-/download.\n", nCh); 
    int device = 1;  // CUDAwDT

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat filterBox = cuCV::ones<CUCV_64F>(3, 3, nCh) / 9;
            cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
            cuCV::CuMat<CUCV_64F> filterBox_dev(filterBox);

            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            filterBox_dev.uploadFrom(filterBox);
            cuCV::CuMat cucvMatResult_dev = cuCV::slowConv2d(cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
            cucvMatResult_dev.downloadTo(cucvMatResult);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on host using openCV
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dOpenCV(int N, int nCh, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on host using openCV.\n", nCh); 
    int device = 5;  // openCV

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cv::Mat opencvMat1 = cv::Mat::ones(n, n, getCVequivalent(nCh));
            cv::Mat opencvMat2(N, N, getCVequivalent(nCh));
            
            auto tic = begin();
            cv::boxFilter(opencvMat1, opencvMat2, -1, cv::Size(3,3));            

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i, %i, %i, %li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}


/**
 * @brief 
 * Log header:
 *  * Device:
 *      * 0 : Host
 *      * 1 : CUDAwDT
 *      * 2 : CIDAwoDT
 *      * 3 : CUDAwDTshared
 *      * 4 : CIDAwoDTshared
 *      * 5 : openCV
 *  * Run
 *  * Pixel: N * N * nCh
 *  * Time in ns
 */
int main(int argc, char ** argv) {
    constexpr int N = 1024 * 2 * 2 * 2;  // Max number of one matrix side.
    int nCh = 1;  // Number of channels
    int I_MAX = 1;  // Perform every computation I_MAX times

    {
        // Measure Addition
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("addition.log");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device [0:Host 1:CUDAwDT 2:CIDAwoDT 3:CUDAwDTshared 4:CIDAwoDTshared 5:openCV], Run [i], N pixel [N * N * %i], Time [ns]\n", nCh);

        additionHost(N, nCh, I_MAX, log);
        additionCUDAwDT(N, nCh, I_MAX, log);
        additionCUDAwoDT(N, nCh, I_MAX, log);
        additionOpenCV(N, nCh, I_MAX, log);

        fclose(log);
    }
    {
        // Measure Matrix Multiplication
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("matmul.log");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device [0:Host 1:CUDAwDT 2:CUDAwoDT 3:CUDAwDTshared 4:CUDAwoDTshared 5:openCV], Run [i], N pixel [N * N * %i], Time [ns]\n", nCh);

        matmulCUDAwDT(N, nCh, I_MAX, log);
        matmulCUDAwoDT(N, nCh, I_MAX, log);
        matmulCUDAwDTshared(N, nCh, I_MAX, log);
        matmulCUDAwoDTshared(N, nCh, I_MAX, log);
        matmulOpenCV(N, nCh, I_MAX, log);

        fclose(log);
    }
    {
        // Measure Convolution
        fs::path fpath = fs::path(__FILE__).parent_path() / "logs" / fs::path("convolution.log");
        FILE * log = fopen(fpath.c_str(), "w");
        fprintf(log, "Device [0:Host 1:CUDAwDT 2:CUDAwoDT 3:CUDAwDTshared 4:CUDAwoDTshared 5:openCV], Run [i], N pixel [N * N * %i], Time [ns]\n", nCh);

        conv2dCUDAwDT(N, nCh, I_MAX, log);
        conv2dOpenCV(N, nCh, I_MAX, log);

        fclose(log);
    }
}
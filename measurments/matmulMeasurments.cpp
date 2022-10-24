#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "linalg.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "measurmentsHelper.cpp"

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
        devWarmup();
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2), 
                    cucvMat3_dev(cucvMat1.getHeight(), cucvMat1.getWidth(), cucvMat1.getNChannels());

            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            cuCV::simpleMatmul(cucvMat3_dev, cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i;%i;%i;%li\n", device, i, n, toc);
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
        devWarmup();
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2), 
                    cucvMat3_dev(cucvMat1.getHeight(), cucvMat1.getWidth(), cucvMat1.getNChannels());

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            
            auto tic = begin();
            cuCV::simpleMatmul(cucvMat3_dev, cucvMat1_dev, cucvMat2_dev);
            auto toc = stop(tic);

            cucvMat3_dev.downloadTo(cucvMat3);
            total += toc;
            fprintf(file, "%i;%i;%i;%li\n", device, i, n, toc);
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
        devWarmup();
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2), 
                    cucvMat3_dev(cucvMat1.getHeight(), cucvMat1.getWidth(), cucvMat1.getNChannels());
            auto tic = begin();

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            cuCV::matmul(cucvMat3_dev, cucvMat1_dev, cucvMat2_dev);
            cucvMat3_dev.downloadTo(cucvMat3);

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i;%i;%i;%li\n", device, i, n, toc);
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
        devWarmup();
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(n, n, nCh);
            cuCV::Mat<CUCV_XX> cucvMat3(n, n, nCh);
            cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2), 
                    cucvMat3_dev(cucvMat1.getHeight(), cucvMat1.getWidth(), cucvMat1.getNChannels());

            cucvMat1_dev.uploadFrom(cucvMat1);
            cucvMat2_dev.uploadFrom(cucvMat2);
            
            auto tic = begin();
            cuCV::matmul(cucvMat3_dev, cucvMat1_dev, cucvMat2_dev);
            auto toc = stop(tic);
            
            cucvMat3_dev.downloadTo(cucvMat3);
            total += toc;
            fprintf(file, "%i;%i;%i;%li\n", device, i, n, toc);
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
    int device = 99;  // openCV

    for (int n = 4; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        opencvWarmup();
        int64_t total = 0;
        for (int i = 0; i < i_max; ++i) {  
            cv::Mat opencvMat1 = cv::Mat::ones(n, n, getCVequivalent(nCh));
            cv::Mat opencvMat2 = cv::Mat::ones(n, n, getCVequivalent(nCh));

            cv::randu(opencvMat1, cv::Scalar(0), cv::Scalar(256));
            
            auto tic = begin();
            cv::Mat opencvMat3 = opencvMat1 * opencvMat2;

            auto toc = stop(tic);
            total += toc;
            fprintf(file, "%i;%i;%i;%li\n", device, i, n, toc);
        }
        total /= i_max;  // results in truncation of int (nanoseconds)!
        fprintf(stdout, "N=%i took %f sec (avg)\n", n, (double) total / 1000000000);
    }
    fprintf(stdout, "\n");
}
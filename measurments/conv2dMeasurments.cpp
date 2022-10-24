#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "linalg.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "measurmentsHelper.cpp"


/**
 * @brief Runtime Measurement of convolution on CUDA device considering upload and download time (data transfer(DT)).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwDT(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device with up-/download.\n", nCh); 
    int device = 1;  // CUDAwDT

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            if (k > n)
                break;
            int64_t total = 0;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_64F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_64F> filterBox_dev(filterBox);

                auto tic = begin();

                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);
                cuCV::simpleConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                cucvMatResult_dev.downloadTo(cucvMatResult);

                auto toc = stop(tic);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%i took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device without considering upload and download time (data transfer(DT)).
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwoDT(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device without up-/download.\n", nCh); 
    int device = 2;  // CUDAwoDT

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_64F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_64F> filterBox_dev(filterBox);

                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);

                auto tic = begin();
                cuCV::simpleConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                auto toc = stop(tic);

                cucvMatResult_dev.downloadTo(cucvMatResult);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%i took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device considering upload and download time (data transfer(DT))
 * and using shared memory.
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwDTshared(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory with up-/download.\n", nCh); 
    int device = 3;  // CUDAwDTshared

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_32F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBox_dev(filterBox);

                auto tic = begin();
                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);
                cuCV::simpleSharedConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                cucvMatResult_dev.downloadTo(cucvMatResult);
                auto toc = stop(tic);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device without considering upload and download time (data transfer(DT))
 * and using shared memory.
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwoDTshared(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory without up-/download.\n", nCh); 
    int device = 4;  // CUDAwoDTshared

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_32F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBox_dev(filterBox);

                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);
                
                auto tic = begin();
                cuCV::simpleSharedConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                auto toc = stop(tic);

                cucvMatResult_dev.downloadTo(cucvMatResult);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device considering upload and download time (data transfer(DT))
 * and using shared memory using method sharedPaddingConv2d().
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwDTsharedPadded(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory using method sharedPaddingConv2d() with up-/download.\n", nCh); 
    int device = 5;  // CUDAwDTsharedPadded

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            if (std::min(BLOCK_SIZE, 16) < k / 2)
                break; // sharedPaddingConv2d not allowed in that case
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_32F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBox_dev(filterBox);

                auto tic = begin();
                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);
                cuCV::sharedPaddingConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                cucvMatResult_dev.downloadTo(cucvMatResult);
                auto toc = stop(tic);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device without considering upload and download time (data transfer(DT))
 * and using shared memory using method sharedPaddingConv2d().
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwoDTsharedPadded(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory using method sharedPaddingConv2d() without up-/download.\n", nCh); 
    int device = 6;  // CUDAwoDTsharedPadded

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            if (std::min(BLOCK_SIZE, 16) < k / 2)
                break; // sharedPaddingConv2d not allowed in that case
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBox = cuCV::ones<CUCV_32F>(k, k, nCh) / (k*k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBox_dev(filterBox);

                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBox_dev.uploadFrom(filterBox);
                
                auto tic = begin();
                cuCV::sharedPaddingConv2d(cucvMatResult_dev, cucvMat1_dev, filterBox_dev, cuCV::Padding::ZERO);
                auto toc = stop(tic);

                cucvMatResult_dev.downloadTo(cucvMatResult);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device considering upload and download time (data transfer(DT))
 * and using shared memory using method sepSharedConv2d().
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwDTsharedSeparated(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory using method sepSharedConv2d() with up-/download.\n", nCh); 
    int device = 7;  // CUDAwDTsharedSeparated

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBoxRow = cuCV::ones<CUCV_32F>(k, 1, nCh) / (k);
                cuCV::Mat filterBoxCol = cuCV::ones<CUCV_32F>(1, k, nCh) / (k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBoxRow_dev(filterBoxRow);
                cuCV::CuMat<CUCV_32F> filterBoxCol_dev(filterBoxCol);

                auto tic = begin();
                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBoxRow_dev.uploadFrom(filterBoxRow);
                filterBoxCol_dev.uploadFrom(filterBoxCol);
                cuCV::sepSharedConv2d(cucvMatResult_dev, cucvMat1_dev, filterBoxRow_dev, filterBoxCol_dev, cuCV::Padding::ZERO);
                cudaDeviceSynchronize();
                cucvMatResult_dev.downloadTo(cucvMatResult);
                auto toc = stop(tic);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on CUDA device without considering upload and download time (data transfer(DT))
 * and using shared memory using method sharedPaddingConv2d().
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dCUDAwoDTsharedSeparated(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on CUDA device using shared memory using method sepSharedConv2d() without up-/download.\n", nCh); 
    int device = 8;  // CUDAwoDTsharedSeparated

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            devWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(n, n, nCh);
                cuCV::Mat filterBoxRow = cuCV::ones<CUCV_32F>(k, 1, nCh) / (k);
                cuCV::Mat filterBoxCol = cuCV::ones<CUCV_32F>(1, k, nCh) / (k);
                cuCV::Mat<CUCV_XX> cucvMatResult(n, n, nCh);
                cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1);
                cuCV::CuMat<CUCV_XX> cucvMatResult_dev(cucvMat1);
                cuCV::CuMat<CUCV_32F> filterBoxRow_dev(filterBoxRow);
                cuCV::CuMat<CUCV_32F> filterBoxCol_dev(filterBoxCol);

                cucvMat1_dev.uploadFrom(cucvMat1);
                filterBoxRow_dev.uploadFrom(filterBoxRow);
                filterBoxCol_dev.uploadFrom(filterBoxCol);
                
                auto tic = begin();
                cuCV::sepSharedConv2d(cucvMatResult_dev, cucvMat1_dev, filterBoxRow_dev, filterBoxCol_dev, cuCV::Padding::ZERO);
                cudaDeviceSynchronize();
                auto toc = stop(tic);

                cucvMatResult_dev.downloadTo(cucvMatResult);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%d, took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}


/**
 * @brief Runtime Measurement of convolution on host using openCV
 * 
 * @param N Length of one side of the matrix.
 * @param nCh Number of channels of matrix.
 * @param kSize Maximum length of one side of the kernel.
 * @param i_max Number of total computations from which an average runtime in calculated.
 * @param file Pointer to the log file stream.
 */
void conv2dOpenCV(int N, int nCh, int kSize, int i_max, FILE * file) {
    fprintf(stdout, "Measuring runtime of convolution with size N * N * %i on host using openCV.\n", nCh); 
    int device = 99;  // openCV

    for (int n = 2048; n <= N; n *= 2) {  // increase dimension logarithmic base 2 up to 2 ** 14
        for (int k = 1; k <= kSize; k += 2) {
            opencvWarmup();
            int64_t total = 0;
            if (k > n)
                break;
            for (int i = 0; i < i_max; ++i) {  
                cv::Mat opencvMat1 = cv::Mat::ones(n, n, getCVequivalent(nCh));
                cv::randu(opencvMat1, cv::Scalar(0), cv::Scalar(256));
                cv::Mat opencvMat2(N, N, getCVequivalent(nCh));
                cv::Mat filterBoxRow = cv::Mat::ones(1, k, CV_32F);
                cv::Mat filterBoxCol = cv::Mat::ones(k, 1, CV_32F);
                
                auto tic = begin();
                cv::sepFilter2D(opencvMat1, opencvMat2, -1, filterBoxRow, filterBoxCol);            

                auto toc = stop(tic);
                total += toc;
                fprintf(file, "%i;%i;%i;%i;%li\n", device, i, n, k, toc);
            }
            total /= i_max;  // results in truncation of int (nanoseconds)!
            fprintf(stdout, "N=%i, k=%i took %f sec (avg)\n", n, k, (double) total / 1000000000);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}
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

#ifndef MEASURMENTS_HELPER_CPP
#define MEASURMENTS_HELPER_CPP


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
    else if (typeid(CUCV_XX) == typeid(CUCV_32F) && nCh == 3) 
        openCVdtype = CV_32FC3;
    else if (typeid(CUCV_XX) == typeid(CUCV_64F) && nCh == 3) 
        openCVdtype = CV_64FC3;
    else if (typeid(CUCV_XX) == typeid(CUCV_8U) && nCh == 1) 
        openCVdtype = CV_8U;
    else if (typeid(CUCV_XX) == typeid(CUCV_16U) && nCh == 1) 
        openCVdtype = CV_16U;
    else if (typeid(CUCV_XX) == typeid(CUCV_32F) && nCh == 1) 
        openCVdtype = CV_32F;
    else if (typeid(CUCV_XX) == typeid(CUCV_64F) && nCh == 1) 
        openCVdtype = CV_64F;
    else {
        fprintf(stderr, "No openCV datatype equivalent found.");
        abort();
    }

    return openCVdtype;
}

void devWarmup() {
    for (int i=0; i<10; i++) {
        cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(1024, 1024, 3);
        cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(1024, 1024, 3);
        cuCV::Mat<CUCV_XX> cucvMat3(1024, 1024, 3);
        cuCV::CuMat<CUCV_XX> cucvMat1_dev(cucvMat1), cucvMat2_dev(cucvMat2);

        cucvMat1_dev.uploadFrom(cucvMat1);
        cucvMat2_dev.uploadFrom(cucvMat2);
        cuCV::CuMat cucvMat3_dev = cucvMat1_dev + cucvMat2_dev;
        cucvMat3_dev.downloadTo(cucvMat3);
    }
}

void opencvWarmup() {
    for (int i=0; i<10; i++) {
        cv::Mat opencvMat1 = cv::Mat::ones(1024, 1024, getCVequivalent(3));
        cv::Mat opencvMat2 = cv::Mat::ones(1024, 1024, getCVequivalent(3));
        cv::Mat opencvMat3 = opencvMat1 + opencvMat2;
    }
}

void hostWarmup() {
    for (int i=0; i<10; i++) {
        cuCV::Mat cucvMat1 = cuCV::ones<CUCV_XX>(1024, 1024, 3);
        cuCV::Mat cucvMat2 = cuCV::ones<CUCV_XX>(1024, 1024, 3);
        cuCV::Mat cucvMat3 = cucvMat1 + cucvMat2;
    }
}

#endif

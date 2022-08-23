/**
 * @file cumatinitializersTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "cumatinitializersTest.h"

// Register Initializers Testing Methods
CPPUNIT_TEST_SUITE_REGISTRATION(CuMatInitializersTest);

void CuMatInitializersTest::setUp() {
    N = 10;
}


void CuMatInitializersTest::testZeros() {
    cuCV::CuMat A_dev = cuCV::zerosOnDevice<CUCV_8U>(N, N, N);
    cuCV::CuMat B_dev = cuCV::zerosOnDevice<CUCV_16U>(N, N, N);
    cuCV::CuMat C_dev = cuCV::zerosOnDevice<CUCV_64F>(N, N, N);

    cuCV::Mat<CUCV_8U> A(N, N, N);
    cuCV::Mat<CUCV_16U> B(N, N, N);
    cuCV::Mat<CUCV_64F> C(N, N, N);

    A_dev.downloadTo(A);
    B_dev.downloadTo(B);
    C_dev.downloadTo(C);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.mData[i]==0);
        CPPUNIT_ASSERT(B.mData[i]==0);
        CPPUNIT_ASSERT(C.mData[i]==0);
    }
}


void CuMatInitializersTest::testOnes() {
    cuCV::CuMat A_dev = cuCV::onesOnDevice<CUCV_8U>(N, N, N);
    cuCV::CuMat B_dev = cuCV::onesOnDevice<CUCV_16U>(N, N, N);
    cuCV::CuMat C_dev = cuCV::onesOnDevice<CUCV_64F>(N, N, N);

    cuCV::Mat<CUCV_8U> A(N, N, N);
    cuCV::Mat<CUCV_16U> B(N, N, N);
    cuCV::Mat<CUCV_64F> C(N, N, N);

    A_dev.downloadTo(A);
    B_dev.downloadTo(B);
    C_dev.downloadTo(C);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.mData[i]==1);
        CPPUNIT_ASSERT(B.mData[i]==1);
        CPPUNIT_ASSERT(C.mData[i]==1);
    }
}


void CuMatInitializersTest::testEye() {
    cuCV::CuMat A_dev = cuCV::eyeOnDevice<CUCV_8U>(N, N, N);
    cuCV::CuMat B_dev = cuCV::eyeOnDevice<CUCV_16U>(N, N, N);
    cuCV::CuMat C_dev = cuCV::eyeOnDevice<CUCV_64F>(N, N, N);

    cuCV::Mat<CUCV_8U> A(N, N, N);
    cuCV::Mat<CUCV_16U> B(N, N, N);
    cuCV::Mat<CUCV_64F> C(N, N, N);

    A_dev.downloadTo(A);
    B_dev.downloadTo(B);
    C_dev.downloadTo(C);

    cuCV::Mat A_gt = cuCV::eye<CUCV_8U>(N, N, N);
    cuCV::Mat B_gt = cuCV::eye<CUCV_16U>(N, N, N);
    cuCV::Mat C_gt = cuCV::eye<CUCV_64F>(N, N, N);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.mData[i]==A_gt.mData[i]);
        CPPUNIT_ASSERT(B.mData[i]==B_gt.getDataPtr()[i]);
        CPPUNIT_ASSERT(C.mData[i]==C_gt.getDataPtr()[i]);
    }
}

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
        CPPUNIT_ASSERT(A.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==0);
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
        CPPUNIT_ASSERT(A.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==1);
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
        CPPUNIT_ASSERT(A.getDataPtr()[i]==A_gt.getDataPtr()[i]);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==B_gt.getDataPtr()[i]);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==C_gt.getDataPtr()[i]);
    }
}


void CuMatInitializersTest::testGauss() {
    N = 7;
    int M = 5;
    double sig = 1;
    int nCh = 4;
    // cuCV::CuMat A_dev = cuCV::gaussOnDevice<CUCV_8U>(N, N, nCh, sig, true);
    // cuCV::CuMat B_dev = cuCV::gaussOnDevice<CUCV_16U>(N, N, nCh, sig, true);
    cuCV::CuMat C_dev = cuCV::gaussOnDevice<CUCV_64F>(N, M, nCh, sig, true);

    cuCV::Mat<CUCV_8U> A(N,M,nCh);
    cuCV::Mat<CUCV_16U> B(N,M,nCh);
    cuCV::Mat<CUCV_64F> C(N,M,nCh);

    // A_dev.downloadTo(A);
    // B_dev.downloadTo(B);
    C_dev.downloadTo(C);

    // cuCV::Mat A_gt = cuCV::gauss<CUCV_64F>(N, nCh, sig, true);
    // cuCV::Mat B_gt = cuCV::gauss<CUCV_64F>(N, nCh, sig, true);
    cuCV::Mat C_gt = cuCV::gauss<CUCV_64F>(N, M, nCh, sig, true);

    for (size_t i=0; i<N*M*nCh; i++) {
        // CPPUNIT_ASSERT(A.getDataPtr()[i]==A_gt.getDataPtr()[i]);
        // CPPUNIT_ASSERT(B.getDataPtr()[i]==B_gt.getDataPtr()[i]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("i = " + std::to_string(i), C_gt.getDataPtr()[i], C.getDataPtr()[i], 0.001);
    }
}

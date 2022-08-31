/**
 * @file linalgTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "linalgTest.h"

// Register Initializers Testing Methods
CPPUNIT_TEST_SUITE_REGISTRATION(LinalgTest);


void LinalgTest::setUp() {
    N = 10;
    nCh = 4;
}


void LinalgTest::testNaiveMatmul() {
    cuCV::Mat A1 = cuCV::eye<CUCV_8U>(N, N, nCh);
    cuCV::Mat B1 = cuCV::eye<CUCV_16U>(N, N, nCh);
    cuCV::Mat C1 = cuCV::eye<CUCV_64F>(N, N, nCh);

    cuCV::Mat A2 = cuCV::eye<CUCV_8U>(N, N, nCh);
    cuCV::Mat B2 = cuCV::eye<CUCV_16U>(N, N, nCh);
    cuCV::Mat C2 = cuCV::eye<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev1(A1);
    cuCV::CuMat<CUCV_16U> B_dev1(B1);
    cuCV::CuMat<CUCV_64F> C_dev1(C1);

    cuCV::CuMat<CUCV_8U> A_dev2(A2);
    cuCV::CuMat<CUCV_16U> B_dev2(B2);
    cuCV::CuMat<CUCV_64F> C_dev2(C2);

    A_dev1.uploadFrom(A1);
    B_dev1.uploadFrom(B1);
    C_dev1.uploadFrom(C1);

    A_dev2.uploadFrom(A1);
    B_dev2.uploadFrom(B1);
    C_dev2.uploadFrom(C1);

    cuCV::CuMat<CUCV_8U> A_dev_out = cuCV::naiveMatmul(A_dev1, A_dev2);
    cuCV::CuMat<CUCV_16U> B_dev_out = cuCV::naiveMatmul(B_dev1, B_dev2);
    cuCV::CuMat<CUCV_64F> C_dev_out = cuCV::naiveMatmul(C_dev1, C_dev2);

    cuCV::Mat<CUCV_8U> A_out(N,N,nCh);
    cuCV::Mat<CUCV_16U> B_out(N,N,nCh);
    cuCV::Mat<CUCV_64F> C_out(N,N,nCh);

    A_dev_out.downloadTo(A_out);
    B_dev_out.downloadTo(B_out);
    C_dev_out.downloadTo(C_out);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A_out.mData[i]==A1.mData[i]);
        CPPUNIT_ASSERT(B_out.mData[i]==B1.mData[i]);
        CPPUNIT_ASSERT(C_out.mData[i]==C1.mData[i]);
    }
}


void LinalgTest::testMatmul() {
    nCh = 1;
    
    cuCV::Mat A1 = cuCV::eye<CUCV_8U>(N, N, nCh);
    cuCV::Mat B1 = cuCV::eye<CUCV_16U>(N, N, nCh);
    cuCV::Mat C1 = cuCV::eye<CUCV_64F>(N, N, nCh);

    cuCV::Mat A2 = cuCV::eye<CUCV_8U>(N, N, nCh);
    cuCV::Mat B2 = cuCV::eye<CUCV_16U>(N, N, nCh);
    cuCV::Mat C2 = cuCV::eye<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev1(A1);
    cuCV::CuMat<CUCV_16U> B_dev1(B1);
    cuCV::CuMat<CUCV_64F> C_dev1(C1);

    cuCV::CuMat<CUCV_8U> A_dev2(A2);
    cuCV::CuMat<CUCV_16U> B_dev2(B2);
    cuCV::CuMat<CUCV_64F> C_dev2(C2);

    A_dev1.uploadFrom(A1);
    B_dev1.uploadFrom(B1);
    C_dev1.uploadFrom(C1);

    A_dev2.uploadFrom(A1);
    B_dev2.uploadFrom(B1);
    C_dev2.uploadFrom(C1);

    cuCV::CuMat<CUCV_8U> A_dev_out = cuCV::matmul(A_dev1, A_dev2);
    cuCV::CuMat<CUCV_16U> B_dev_out = cuCV::matmul(B_dev1, B_dev2);
    cuCV::CuMat<CUCV_64F> C_dev_out = cuCV::matmul(C_dev1, C_dev2);

    cuCV::Mat<CUCV_8U> A_out(N,N,nCh);
    cuCV::Mat<CUCV_16U> B_out(N,N,nCh);
    cuCV::Mat<CUCV_64F> C_out(N,N,nCh);

    A_dev_out.downloadTo(A_out);
    B_dev_out.downloadTo(B_out);
    C_dev_out.downloadTo(C_out);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A_out.mData[i]==A1.mData[i]);
        CPPUNIT_ASSERT(B_out.mData[i]==B1.mData[i]);
        CPPUNIT_ASSERT(C_out.mData[i]==C1.mData[i]);
    }
}


void LinalgTest::testSimpleConv2d() {
    nCh = 3;
    N = 100;
    int f = 9;
    
    cuCV::Mat A1 = cuCV::eye<CUCV_8U>(N, N, nCh) * f;
    cuCV::Mat B1 = cuCV::eye<CUCV_16U>(N, N, nCh) * f;
    cuCV::Mat C1 = cuCV::eye<CUCV_64F>(N, N, nCh) * f;

    int filterN = 3;

    cuCV::Mat box = cuCV::ones<CUCV_64F>(filterN, filterN, nCh) / 9;

    cuCV::CuMat<CUCV_8U> A_dev1(A1);
    cuCV::CuMat<CUCV_16U> B_dev1(B1);
    cuCV::CuMat<CUCV_64F> C_dev1(C1);

    cuCV::CuMat<CUCV_64F> box_dev(box);

    A_dev1.uploadFrom(A1);
    B_dev1.uploadFrom(B1);
    C_dev1.uploadFrom(C1);

    box_dev.uploadFrom(box);

    cuCV::CuMat<CUCV_8U> A_dev_out = cuCV::simpleConv2d(A_dev1, box_dev, cuCV::Padding::ZERO);
    cuCV::CuMat<CUCV_16U> B_dev_out = cuCV::simpleConv2d(B_dev1, box_dev, cuCV::Padding::ZERO);
    cuCV::CuMat<CUCV_64F> C_dev_out = cuCV::simpleConv2d(C_dev1, box_dev, cuCV::Padding::ZERO);

    cuCV::Mat<CUCV_8U> A_out(N,N,nCh);
    cuCV::Mat<CUCV_16U> B_out(N,N,nCh);
    cuCV::Mat<CUCV_64F> C_out(N,N,nCh);

    A_dev_out.downloadTo(A_out);
    B_dev_out.downloadTo(B_out);
    C_dev_out.downloadTo(C_out);

    for (int ch = 0; ch < nCh; ch++) {
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                if (col == row && row != 0 && row != N-1) {
                    CPPUNIT_ASSERT(A_out.at(row, col, ch) == f / 3);  // main diagonal
                    CPPUNIT_ASSERT(B_out.at(row, col, ch) == f / 3);  // main diagonal
                    CPPUNIT_ASSERT(C_out.at(row, col, ch) == f / 3);  // main diagonal
                }
                else if (abs(col - row) == 1) {
                    CPPUNIT_ASSERT(A_out.at(row, col, ch) == f / 9 * 2);  // first off-diagonal
                    CPPUNIT_ASSERT(B_out.at(row, col, ch) == f / 9 * 2);  // first off-diagonal
                    CPPUNIT_ASSERT(C_out.at(row, col, ch) == f / 9 * 2);  // first off-diagonal
                }
                else if (abs(col - row) == 2) {
                    CPPUNIT_ASSERT(A_out.at(row, col, ch) == f / 9);  // second off-diagonal
                    CPPUNIT_ASSERT(B_out.at(row, col, ch) == f / 9);  // second off-diagonal
                    CPPUNIT_ASSERT(C_out.at(row, col, ch) == f / 9);  // second off-diagonal
                }
            }
        }
    }
    gpuErrchk(cudaDeviceReset());  // to detect leaks with cuda-memcheck
}
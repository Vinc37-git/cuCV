/**
 * @file initializersTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "initializersTest.h"

// Register Initializers Testing Methods
CPPUNIT_TEST_SUITE_REGISTRATION(InitializersTest);

void InitializersTest::setUp() {
    N = 10;
}

void InitializersTest::testZeros() {
    cuCV::Mat A = cuCV::zeros<CUCV_8U>(N, N, N);
    cuCV::Mat B = cuCV::zeros<CUCV_16U>(N, N, N);
    cuCV::Mat C = cuCV::zeros<CUCV_64F>(N, N, N);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.mData[i]==0);
        CPPUNIT_ASSERT(B.mData[i]==0);
        CPPUNIT_ASSERT(C.mData[i]==0);
    }
}


void InitializersTest::testOnes() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, N);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, N);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, N);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.mData[i]==1);
        CPPUNIT_ASSERT(B.mData[i]==1);
        CPPUNIT_ASSERT(C.mData[i]==1);
    }
}


void InitializersTest::testEye() {
    N = 2;

    cuCV::Mat A = cuCV::eye<CUCV_8U>(N, N, N);
    cuCV::Mat B = cuCV::eye<CUCV_16U>(N, N, N);
    cuCV::Mat C = cuCV::eye<CUCV_64F>(N, N, N);

    CUCV_64F testData[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    for (size_t i=0; i<8; i++) {
        CPPUNIT_ASSERT(A.mData[i]==testData[i]);
        CPPUNIT_ASSERT(B.mData[i]==testData[i]);
        CPPUNIT_ASSERT(C.mData[i]==testData[i]);
    }
}


static double discreteGaussianTest(size_t len, int x, int y, double sigma) {
    x -= len/2, y -= len/2;
    double r = x * x + y * y, s = 2.f * sigma * sigma;
    return exp(-r / s ) / (s * M_PI);
}


void InitializersTest::testGauss() {

    N = 5;

    for (double sig = 0.5; sig < 5; sig += 0.4) {
        // cuCV::Mat A = cuCV::gauss<CUCV_8U>(N, N, sig, false);
        // cuCV::Mat B = cuCV::gauss<CUCV_16U>(N, N, sig, false);
        cuCV::Mat C = cuCV::gauss<CUCV_64F>(N, N, sig);

        double scale64F = C.at((int) N/2, (int) N/2) / discreteGaussianTest(N, (int) N/2, (int) N/2, sig);

        for (size_t row = 0; row < N; ++row) {
            for (size_t col = 0; col < N; ++col) {
                // CPPUNIT_ASSERT_EQUAL(A.mData[row * N + col], (unsigned char) discreteGaussianTest(N, col, row, sig));
                // CPPUNIT_ASSERT_EQUAL(B.mData[row * N + col], (unsigned short) discreteGaussianTest(N, col, row, sig));
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(" ", discreteGaussianTest(N, col, row, sig) * scale64F, C.mData[row * N + col], 0.001);
            }
        }
    }

}
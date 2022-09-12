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
        CPPUNIT_ASSERT(A.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==0);
    }
}


void InitializersTest::testOnes() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, N);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, N);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, N);

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==1);
    }
}


void InitializersTest::testEye() {
    N = 2;

    cuCV::Mat A = cuCV::eye<CUCV_8U>(N, N, N);
    cuCV::Mat B = cuCV::eye<CUCV_16U>(N, N, N);
    cuCV::Mat C = cuCV::eye<CUCV_64F>(N, N, N);

    CUCV_64F testData[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    for (size_t i=0; i<8; i++) {
        CPPUNIT_ASSERT(A.getDataPtr()[i]==testData[i]);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==testData[i]);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==testData[i]);
    }
}


static double discreteGaussianTest(int width, int height, int x, int y, double sigma) {
    x -= width/2, y -= height/2;
    double r = x * x + y * y, s = 2.f * sigma * sigma;
    return exp(-r / s ); // / (s * M_PI);
}


void InitializersTest::testGauss() {

    N = 5;
    int M = 9;

    for (double sig = 0.5; sig < 5; sig += 0.4) {
        // cuCV::Mat A = cuCV::gauss<CUCV_8U>(N, N, sig, false);
        // cuCV::Mat B = cuCV::gauss<CUCV_16U>(N, N, sig, false);
        cuCV::Mat C = cuCV::gauss<CUCV_64F>(N, M, N, sig, false);

        for (int ch = 0; ch < N; ++ch) 
                for (int row = 0; row < M; ++row) {
                    for (int col = 0; col < N; ++col) {
                    // CPPUNIT_ASSERT_EQUAL(A.getDataPtr()[row * N + col], (unsigned char) discreteGaussianTest(N, col, row, sig));
                    // CPPUNIT_ASSERT_EQUAL(B.getDataPtr()[row * N + col], (unsigned short) discreteGaussianTest(N, col, row, sig));
                    std::string msg = "Sigma: " + std::to_string(sig) + " row: " + std::to_string(row) + " col: " + std::to_string(col);
                    CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(msg, discreteGaussianTest(N, M, col, row, sig), C.at(row, col, ch), 0.001);
                }
            }
    }

}
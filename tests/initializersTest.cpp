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

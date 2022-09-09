/**
 * @file matTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#include "matTest.h"


// Register Mat Testing Methods
CPPUNIT_TEST_SUITE_REGISTRATION(MatTest);

void MatTest::setUp() {
    N = 10;
    A8U = cuCV::Mat<CUCV_8U>(N, N, N);
    B8U = cuCV::Mat<CUCV_8U>(N, N, N);
    C8U = cuCV::Mat<CUCV_8U>(N, N, N);

    A16U = cuCV::Mat<CUCV_16U>(N, N, N);
    B16U = cuCV::Mat<CUCV_16U>(N, N, N);
    C16U = cuCV::Mat<CUCV_16U>(N, N, N);

    A64F = cuCV::Mat<CUCV_64F>(N, N, N);
    B64F = cuCV::Mat<CUCV_64F>(N, N, N);
    C64F = cuCV::Mat<CUCV_64F>(N, N, N);
}


void MatTest::testZeros() {

    A8U.zeros();
    A16U.zeros();
    A64F.zeros();

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==0);
    }
}


void MatTest::testOnes() {
    A8U.ones();
    A16U.ones();
    A64F.ones();

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==1);
    }
}


void MatTest::testEye() {
    N = 2;

    cuCV::Mat<CUCV_8U> A8U_eye(N, N, N);
    cuCV::Mat<CUCV_16U> A16U_eye(N, N, N);
    cuCV::Mat<CUCV_64F> A64F_eye(N, N, N);

    A8U_eye.eye();
    A16U_eye.eye();
    A64F_eye.eye();

    CUCV_64F testData[8] = {1, 0, 0, 1, 1, 0, 0, 1};

    for (size_t i=0; i<8; i++) {
        CPPUNIT_ASSERT(A8U_eye.getDataPtr()[i]==testData[i]);
        CPPUNIT_ASSERT(A16U_eye.getDataPtr()[i]==testData[i]);
        CPPUNIT_ASSERT(A64F_eye.getDataPtr()[i]==testData[i]);
    }
}


void MatTest::testAdd() {
    A8U.ones();
    A16U.ones();
    A64F.ones();

    A8U += 3;
    A16U += 4;
    A64F += 5;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==4);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i]==5);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==6);
    }

    // Add and assign to self
    A8U = A8U + A8U;

    // Add different Mats
    B16U.ones();
    C16U = A16U + B16U;

    // Add multiple Mats in one line
    B64F.ones(); C64F.ones();
    A64F = B64F + B64F + C64F + C64F;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==8);
        CPPUNIT_ASSERT(C16U.getDataPtr()[i]==6);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==4);
    }
}


void MatTest::testMul() {
    A8U.ones();
    A16U.ones();
    A64F.ones();

    A8U *= 3;
    A16U *= 4;
    A64F *= 0.5;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==3);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i]==4);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==0.5);
    }

    // Multiply and assign to self
    A8U = A8U * A8U;  // 3 * 3

    // Multiply different Mats
    B16U.ones();
    B16U += 4;
    C16U = A16U * B16U;  // 4 * 5

    // Add multiple Mats in one line
    B64F.ones(); C64F.ones();
    B64F *= 0.7; C64F *= 0.2;
    A64F = B64F * B64F * C64F * C64F;  // 0.7 * 0.7 * 0.2 * 0.2

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==9);
        CPPUNIT_ASSERT(C16U.getDataPtr()[i]==20);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==0.7 * 0.7 * 0.2 * 0.2);
    }
}


void MatTest::testDif() {
    A8U.zeros();
    A16U.zeros();
    A64F.ones();
    
    A8U += CUCV_8U_MAX;
    A16U += CUCV_16U_MAX;

    A8U -= 127;
    A16U -= 32767;
    A64F -= 0.3;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==128);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i]==32768);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==0.7);
    }

    // Add and assign to self
    A8U = A8U - A8U;

    // Add different Mats
    B16U.ones();
    B16U *= 2768;
    C16U = A16U - B16U;

    // Add multiple Mats in one line
    B64F.ones(); C64F.ones();
    B64F *= 0.1; C64F *= 0.7;
    A64F = B64F - B64F - C64F - C64F;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==0);
        CPPUNIT_ASSERT(C16U.getDataPtr()[i]==30000);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==-1.4);
    }
}


void MatTest::testDiv() {
    A8U.zeros();
    A16U.zeros();
    A64F.ones();
    
    A8U += CUCV_8U_MAX;
    A16U += CUCV_16U_MAX;
    A16U -= 0.2;

    A8U /= 2;
    A16U /= 2;
    A64F /= 2;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==127);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i]==32767);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==0.5);
    }

    // Divide by itself and assign to self
    A8U = A8U / A8U;

    // Divide different Mats
    B16U.ones();
    B16U *= 3;
    C16U = A16U / B16U;

    // Divide multiple Mats in one line
    B64F.ones(); C64F.ones();
    B64F *= 0.1; C64F *= 0.7;
    A64F = A64F / B64F / B64F / C64F / C64F;

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(C16U.getDataPtr()[i]==32767/3);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i]==0.5 / 0.1 / 0.1 / 0.7 / 0.7);
    }
}


void MatTest::testConvert() {
    A8U.ones();
    A16U.ones();
    A64F.ones();

    // A 8U to 16U
    cuCV::Mat A8U_to_A16U = A8U.astype<CUCV_16U>();
    cuCV::Mat A16U_to_A64F = A8U.astype<CUCV_64F>();
    cuCV::Mat A64F_to_A8U = A8U.astype<CUCV_8U>();

    cuCV::Mat A8U_to_A64F = A8U.astype<CUCV_64F>();
    cuCV::Mat A16U_to_A8U = A8U.astype<CUCV_8U>();
    cuCV::Mat A64F_to_A16U = A8U.astype<CUCV_16U>();

    for (size_t i=0; i<N*N*N; i++) {
        CPPUNIT_ASSERT(A8U.getDataPtr()[i] == A64F_to_A8U.getDataPtr()[i]);
        CPPUNIT_ASSERT(A8U.getDataPtr()[i] == A16U_to_A8U.getDataPtr()[i]);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i] == A8U_to_A16U.getDataPtr()[i]);
        CPPUNIT_ASSERT(A16U.getDataPtr()[i] == A64F_to_A16U.getDataPtr()[i]);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i] == A8U_to_A64F.getDataPtr()[i]);
        CPPUNIT_ASSERT(A64F.getDataPtr()[i] == A16U_to_A64F.getDataPtr()[i]);
    }
}
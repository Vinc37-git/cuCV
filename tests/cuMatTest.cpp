/**
 * @file cuMatTest.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "cuMatTest.h"

// Register Initializers Testing Methods
CPPUNIT_TEST_SUITE_REGISTRATION(CuMatTest);


void CuMatTest::setUp() {
    N = 10;
    nCh = 4;
}


void CuMatTest::testUpload() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, nCh);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, nCh);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_16U> B_dev(B);
    cuCV::CuMat<CUCV_64F> C_dev(C);

    // Device pointers should be zero.
    CPPUNIT_ASSERT(A_dev.getDataPtr()==NULL);
    CPPUNIT_ASSERT(B_dev.getDataPtr()==NULL);
    CPPUNIT_ASSERT(C_dev.getDataPtr()==NULL);

    // Send data to device
    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);
    C_dev.uploadFrom(C);

    CPPUNIT_ASSERT(A_dev.getDataPtr()!=NULL);
    CPPUNIT_ASSERT(B_dev.getDataPtr()!=NULL);
    CPPUNIT_ASSERT(C_dev.getDataPtr()!=NULL);
}


void CuMatTest::testUpDownload() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, nCh);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, nCh);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_16U> B_dev(B);
    cuCV::CuMat<CUCV_64F> C_dev(C);

    // Device pointers should be zero.
    CPPUNIT_ASSERT(A_dev.getDataPtr()==NULL);
    CPPUNIT_ASSERT(B_dev.getDataPtr()==NULL);
    CPPUNIT_ASSERT(C_dev.getDataPtr()==NULL);

    // Send data to device
    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);
    C_dev.uploadFrom(C);

    CPPUNIT_ASSERT(A_dev.getDataPtr()!=NULL);
    CPPUNIT_ASSERT(B_dev.getDataPtr()!=NULL);
    CPPUNIT_ASSERT(C_dev.getDataPtr()!=NULL);

    // Download data to host
    cuCV::Mat A1 = cuCV::Mat<CUCV_8U>(N, N, nCh);
    cuCV::Mat B1 = cuCV::Mat<CUCV_16U>(N, N, nCh);
    cuCV::Mat C1 = cuCV::Mat<CUCV_64F>(N, N, nCh);

    A_dev.downloadTo(A1);
    B_dev.downloadTo(B1);
    C_dev.downloadTo(C1);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A1.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(B1.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(C1.getDataPtr()[i]==1);
    }

    // Download data to new uninitialised Matrix
    cuCV::Mat<CUCV_8U> A2;
    cuCV::Mat<CUCV_16U> B2;
    cuCV::Mat<CUCV_64F> C2;

    CPPUNIT_ASSERT_NO_THROW(A_dev.downloadTo(A2));
    CPPUNIT_ASSERT_NO_THROW(B_dev.downloadTo(B2));
    CPPUNIT_ASSERT_NO_THROW(C_dev.downloadTo(C2));

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A2.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(B2.getDataPtr()[i]==1);
        CPPUNIT_ASSERT(C2.getDataPtr()[i]==1);
    }
}


void CuMatTest::testAssignment() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, nCh) + 1;
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, nCh) + 2;
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, nCh) + 3;

    cuCV::CuMat<CUCV_8U> A_dev(A), A_dev_copy;
    cuCV::CuMat<CUCV_16U> B_dev(B), B_dev_copy;
    cuCV::CuMat<CUCV_64F> C_dev(C), C_dev_copy;

    // Send data to device
    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);
    C_dev.uploadFrom(C);

    A_dev_copy = A_dev;
    B_dev_copy = B_dev;
    C_dev_copy = C_dev;

    CPPUNIT_ASSERT(A_dev.getDataPtr()!=A_dev_copy.getDataPtr());
    CPPUNIT_ASSERT(B_dev.getDataPtr()!=B_dev_copy.getDataPtr());
    CPPUNIT_ASSERT(C_dev.getDataPtr()!=C_dev_copy.getDataPtr());

    cuCV::Mat<CUCV_8U> A_copy_from_device(N,N,nCh);
    cuCV::Mat<CUCV_16U> B_copy_from_device(N,N,nCh);
    cuCV::Mat<CUCV_64F> C_copy_from_device(N,N,nCh);

    // Download data to host
    A_dev_copy.downloadTo(A_copy_from_device);
    B_dev_copy.downloadTo(B_copy_from_device);
    C_dev_copy.downloadTo(C_copy_from_device);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A.getDataPtr()[i]==A_copy_from_device.getDataPtr()[i]);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==B_copy_from_device.getDataPtr()[i]);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==C_copy_from_device.getDataPtr()[i]);
    }
}


void CuMatTest::testDimMismatch() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(10, 5, 2);
    cuCV::Mat B = cuCV::ones<CUCV_8U>(10, 15, 3);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_8U> B_dev(B);

    CPPUNIT_ASSERT_THROW(A_dev.uploadFrom(B);, cuCV::exception::DimensionMismatch<CUCV_8U>);

    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);

    CPPUNIT_ASSERT_THROW(A_dev += B_dev, cuCV::exception::DimensionMismatch<CUCV_8U>);

    A_dev += A_dev;

    CPPUNIT_ASSERT_THROW(A_dev.downloadTo(B);, cuCV::exception::DimensionMismatch<CUCV_8U>);
}


void CuMatTest::testEmpty() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(10, 5, 2);
    cuCV::Mat B = cuCV::ones<CUCV_8U>(10, 15, 3);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_8U> B_dev(B);

    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);

    B_dev.clearOnDevice();

    CPPUNIT_ASSERT_THROW(A_dev += B_dev, cuCV::exception::NullPointer);
}


void CuMatTest::testAdd() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, nCh);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, nCh);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_16U> B_dev(B);
    cuCV::CuMat<CUCV_64F> C_dev(C);

    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);
    C_dev.uploadFrom(C);

    A_dev += 3;
    B_dev += 4;
    C_dev += 5;

    // Download data to host
    cuCV::Mat A1 = cuCV::Mat<CUCV_8U>(N, N, nCh);
    cuCV::Mat B1 = cuCV::Mat<CUCV_16U>(N, N, nCh);
    cuCV::Mat C1 = cuCV::Mat<CUCV_64F>(N, N, nCh);

    A_dev.downloadTo(A1);
    B_dev.downloadTo(B1);
    C_dev.downloadTo(C1);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A1.getDataPtr()[i]==4);
        CPPUNIT_ASSERT(B1.getDataPtr()[i]==5);
        CPPUNIT_ASSERT(C1.getDataPtr()[i]==6);
    }

    // Add and assign to self
    A_dev = A_dev + A_dev;  // 4 + 4

    // Add different Mats
    cuCV::CuMat<CUCV_16U> B_dev2 = B_dev + 6; // 5 + 6
    cuCV::CuMat<CUCV_16U> B_dev3 = B_dev + B_dev2;  // 5 + 11

    // Add multiple Mats in one line
    cuCV::CuMat<CUCV_64F> C_dev2, C_dev3, C_dev4, C_dev5;
    C_dev2 = C_dev + 10;  //  6 + 10
    C_dev3 = C_dev + 13;  //  6 + 13
    C_dev4 = C_dev + 1;  //  6 + 1
    C_dev5 = C_dev + C_dev2 + C_dev3 + C_dev4;  // 6 + 16 + 19 + 7 = 48

    A_dev.downloadTo(A1);
    B_dev3.downloadTo(B1);
    C_dev5.downloadTo(C1);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A1.getDataPtr()[i]==8);
        CPPUNIT_ASSERT(B1.getDataPtr()[i]==16);
        CPPUNIT_ASSERT(C1.getDataPtr()[i]==48);
    }
}


void CuMatTest::testDifMulDiv() {
    cuCV::Mat A = cuCV::ones<CUCV_8U>(N, N, nCh);
    cuCV::Mat B = cuCV::ones<CUCV_16U>(N, N, nCh);
    cuCV::Mat C = cuCV::ones<CUCV_64F>(N, N, nCh);

    cuCV::CuMat<CUCV_8U> A_dev(A);
    cuCV::CuMat<CUCV_16U> B_dev(B);
    cuCV::CuMat<CUCV_64F> C_dev(C);

    A_dev.uploadFrom(A);
    B_dev.uploadFrom(B);
    C_dev.uploadFrom(C);

    A_dev *= 3;
    B_dev *= 4;
    C_dev *= 0.5;

    //test
    A_dev.downloadTo(A);
    B_dev.downloadTo(B);
    C_dev.downloadTo(C);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A.getDataPtr()[i]==3);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==4);
        CPPUNIT_ASSERT(C.getDataPtr()[i]==0.5);
    }

    cuCV::CuMat<CUCV_8U> A_dev2 = (A_dev - (A_dev/2)) * 10;  // (3 - (3/2)) * 10 = 20
    cuCV::CuMat<CUCV_16U> B_dev2 = (B_dev - (B_dev/2)) * 10;  // (4 - (4/2)) * 10 = 20
    cuCV::CuMat<CUCV_64F> C_dev2 = (C_dev + 0.1) / 2 - 0.2;  // (0.5+0.1) / 2 - 0.2 = 0.1

    //test
    A_dev2.downloadTo(A);
    B_dev2.downloadTo(B);
    C_dev2.downloadTo(C);

    for (size_t i=0; i<N*N*nCh; i++) {
        CPPUNIT_ASSERT(A.getDataPtr()[i]==20);
        CPPUNIT_ASSERT(B.getDataPtr()[i]==20);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, C.getDataPtr()[i], 0.001);
    }



}

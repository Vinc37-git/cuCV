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

void LinalgTest::tearDown() {
    gpuErrchk(cudaDeviceReset());
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
        CPPUNIT_ASSERT(A_out.getDataPtr()[i]==A1.getDataPtr()[i]);
        CPPUNIT_ASSERT(B_out.getDataPtr()[i]==B1.getDataPtr()[i]);
        CPPUNIT_ASSERT(C_out.getDataPtr()[i]==C1.getDataPtr()[i]);
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
        CPPUNIT_ASSERT(A_out.getDataPtr()[i]==A1.getDataPtr()[i]);
        CPPUNIT_ASSERT(B_out.getDataPtr()[i]==B1.getDataPtr()[i]);
        CPPUNIT_ASSERT(C_out.getDataPtr()[i]==C1.getDataPtr()[i]);
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
}




void LinalgTest::testSepConv2d() {
    nCh = 1;
    N = 99;
    int sigma = N/2;
    int kSize = 3;
    int f = 100;

    // Other datatypes than CUCV_32F are tested in testAllConv2d
    cuCV::CuMat<CUCV_32F> A = cuCV::gaussOnDevice<CUCV_32F>(N,N,nCh,sigma,true) * f;
    cuCV::CuMat<CUCV_32F> out0 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> out1 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> out2 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> out3 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> t0 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> t1 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> t2 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> t3 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
    cuCV::CuMat<CUCV_32F> kernel = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, kSize);
    cuCV::CuMat<CUCV_32F> k_row = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, 1);
    cuCV::CuMat<CUCV_32F> k_col = cuCV::createKernel(cuCV::Kernel::GAUSS, 1, kSize);
    cuCV::CuMat<CUCV_32F> dot = cuCV::createKernel(cuCV::Kernel::GAUSS, 1, 1);

    cuCV::simpleConv2d(t0, A, dot, cuCV::Padding::ZERO);
    cuCV::simpleConv2d(t1, A, k_row, cuCV::Padding::ZERO);
    cuCV::simpleConv2d(t2, A, k_col, cuCV::Padding::ZERO);
    cuCV::simpleConv2d(t3, A, kernel, cuCV::Padding::ZERO);

    cuCV::sepSharedConv2d(out0, A, dot, dot, cuCV::Padding::ZERO);  // should do nothing
    cuCV::sepSharedConv2d(out1, A, k_row, dot, cuCV::Padding::ZERO);  // test sepRowConv2d kernel
    cuCV::sepSharedConv2d(out2, A, dot, k_col, cuCV::Padding::ZERO);  // test sepColConv2d kernel
    cuCV::sepSharedConv2d(out3, A, k_row, k_col, cuCV::Padding::ZERO);  // test both kernels

    cuCV::Mat<CUCV_32F> t0_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> t1_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> t2_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> t3_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> out0_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> out1_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> out2_h(N,N,nCh);
    cuCV::Mat<CUCV_32F> out3_h(N,N,nCh);

    t0.downloadTo(t0_h);
    t1.downloadTo(t1_h);
    t2.downloadTo(t2_h);
    t3.downloadTo(t3_h);
    out0.downloadTo(out0_h);
    out1.downloadTo(out1_h);
    out2.downloadTo(out2_h);
    out3.downloadTo(out3_h);

    cuCV::Mat<CUCV_32F> t0_h_master = cuCV::gauss<CUCV_32F>(N,N,nCh,sigma,true) * f;

    for (int ch = 0; ch < nCh; ch++) {
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                std::stringstream rowCol; rowCol << "CUCV_32F - " << "R: "<< row <<" C: " << col ;
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), t0_h.at(row, col, ch), t0_h_master.at(row, col, ch), 0.001);  // is simpleConv2d wporking properly with kernel of size 1?
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), t0_h.at(row, col, ch), out0_h.at(row, col, ch), 0.001);  // is sepSharedConv2d working correct when kernel size is 1;
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), t1_h.at(row, col, ch), out1_h.at(row, col, ch), 0.001); // is only sepRowConv2d correct? 
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), t2_h.at(row, col, ch), out2_h.at(row, col, ch), 0.001);  //  is only sepColConv2d correct?
                CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), t3_h.at(row, col, ch), out3_h.at(row, col, ch), 0.001);  //is both together in sepSharedConv2d working correct
            }
        }
    }
}




void LinalgTest::testAllConv2d() {
    nCh = 1;
    N = 99;
    int sigma = N/2;
    int kSize = 3;
    int f = 100;
    
    for (kSize = 3; kSize < 50; kSize += 6) {
        {  // for CUCV_8U
            cuCV::CuMat<CUCV_8U> A = cuCV::eyeOnDevice<CUCV_8U>(N,N,nCh);
            cuCV::CuMat<CUCV_8U> A_out_target = cuCV::zerosOnDevice<CUCV_8U>(N, N, nCh);
            cuCV::CuMat<CUCV_8U> A_out_calc1 = cuCV::zerosOnDevice<CUCV_8U>(N, N, nCh);
            cuCV::CuMat<CUCV_8U> A_out_calc2 = cuCV::zerosOnDevice<CUCV_8U>(N, N, nCh);
            cuCV::CuMat<CUCV_8U> A_out_calc3 = cuCV::zerosOnDevice<CUCV_8U>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> kernel = cuCV::createKernel(cuCV::Kernel::BOX, kSize, kSize);
            cuCV::CuMat<CUCV_32F> k_row = cuCV::createKernel(cuCV::Kernel::BOX, kSize, 1);
            cuCV::CuMat<CUCV_32F> k_col = cuCV::createKernel(cuCV::Kernel::BOX, 1, kSize);
            cuCV::simpleConv2d(A_out_target, A, kernel, cuCV::Padding::ZERO);
            cuCV::simpleSharedConv2d(A_out_calc1, A, kernel, cuCV::Padding::ZERO);
            if (kSize / 2 < 16) 
                cuCV::sharedPaddingConv2d(A_out_calc2, A, kernel, cuCV::Padding::ZERO);
            cuCV::sepSharedConv2d(A_out_calc3, A, k_row, k_col, cuCV::Padding::ZERO);
            cuCV::Mat<CUCV_8U> A_target_h(N,N,nCh);
            cuCV::Mat<CUCV_8U> A_calc1_h(N,N,nCh);
            cuCV::Mat<CUCV_8U> A_calc2_h(N,N,nCh);
            cuCV::Mat<CUCV_8U> A_calc3_h(N,N,nCh);
            A_out_target.downloadTo(A_target_h);
            A_out_calc1.downloadTo(A_calc1_h);
            A_out_calc2.downloadTo(A_calc2_h);
            A_out_calc3.downloadTo(A_calc3_h);

            for (int ch = 0; ch < nCh; ch++) {
                for (int row = 0; row < N; row++) {
                    for (int col = 0; col < N; col++) {
                        std::stringstream rowCol; rowCol << "CUCV_8U - " << "R: "<< row <<" C: " << col << " - kSize: " << kSize;
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc1_h.at(row, col, ch), 0.001);
                        if (kSize / 2 < 16) 
                            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc2_h.at(row, col, ch), 0.001);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc3_h.at(row, col, ch), 0.001);
                    }
                }
            }
        }
        {  // for CUCV_16U
            cuCV::CuMat<CUCV_16U> A = cuCV::eyeOnDevice<CUCV_16U>(N,N,nCh);
            cuCV::CuMat<CUCV_16U> A_out_target = cuCV::zerosOnDevice<CUCV_16U>(N, N, nCh);
            cuCV::CuMat<CUCV_16U> A_out_calc1 = cuCV::zerosOnDevice<CUCV_16U>(N, N, nCh);
            cuCV::CuMat<CUCV_16U> A_out_calc2 = cuCV::zerosOnDevice<CUCV_16U>(N, N, nCh);
            cuCV::CuMat<CUCV_16U> A_out_calc3 = cuCV::zerosOnDevice<CUCV_16U>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> kernel = cuCV::createKernel(cuCV::Kernel::BOX, kSize, kSize);
            cuCV::CuMat<CUCV_32F> k_row = cuCV::createKernel(cuCV::Kernel::BOX, kSize, 1);
            cuCV::CuMat<CUCV_32F> k_col = cuCV::createKernel(cuCV::Kernel::BOX, 1, kSize);
            cuCV::simpleConv2d(A_out_target, A, kernel, cuCV::Padding::ZERO);
            cuCV::simpleSharedConv2d(A_out_calc1, A, kernel, cuCV::Padding::ZERO);
            if (kSize / 2 < 16) 
                cuCV::sharedPaddingConv2d(A_out_calc2, A, kernel, cuCV::Padding::ZERO);
            cuCV::sepSharedConv2d(A_out_calc3, A, k_row, k_col, cuCV::Padding::ZERO);
            cuCV::Mat<CUCV_16U> A_target_h(N,N,nCh);
            cuCV::Mat<CUCV_16U> A_calc1_h(N,N,nCh);
            cuCV::Mat<CUCV_16U> A_calc2_h(N,N,nCh);
            cuCV::Mat<CUCV_16U> A_calc3_h(N,N,nCh);
            A_out_target.downloadTo(A_target_h);
            A_out_calc1.downloadTo(A_calc1_h);
            A_out_calc2.downloadTo(A_calc2_h);
            A_out_calc3.downloadTo(A_calc3_h);

            for (int ch = 0; ch < nCh; ch++) {
                for (int row = 0; row < N; row++) {
                    for (int col = 0; col < N; col++) {
                        std::stringstream rowCol; rowCol << "CUCV_16U - " << "R: "<< row <<" C: " << col << " - kSize: " << kSize;
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc1_h.at(row, col, ch), 0.001);
                        if (kSize / 2 < 16) 
                            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc2_h.at(row, col, ch), 0.001);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc3_h.at(row, col, ch), 0.001);
                    }
                }
            }
        }
        {  // for CUCV_32F
            cuCV::CuMat<CUCV_32F> A = cuCV::gaussOnDevice<CUCV_32F>(N,N,nCh,sigma,true) * f;
            cuCV::CuMat<CUCV_32F> A_out_target = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> A_out_calc1 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> A_out_calc2 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> A_out_calc3 = cuCV::zerosOnDevice<CUCV_32F>(N, N, nCh);
            cuCV::CuMat<CUCV_32F> kernel = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, kSize);
            cuCV::CuMat<CUCV_32F> k_row = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, 1);
            cuCV::CuMat<CUCV_32F> k_col= cuCV::createKernel(cuCV::Kernel::GAUSS, 1, kSize);
            cuCV::simpleConv2d(A_out_target, A, kernel, cuCV::Padding::ZERO);
            cuCV::simpleSharedConv2d(A_out_calc1, A, kernel, cuCV::Padding::ZERO);
            if (kSize / 2 < 16) 
                cuCV::sharedPaddingConv2d(A_out_calc2, A, kernel, cuCV::Padding::ZERO);
            cuCV::sepSharedConv2d(A_out_calc3, A, k_row, k_col, cuCV::Padding::ZERO);
            cuCV::Mat<CUCV_32F> A_target_h(N,N,nCh);
            cuCV::Mat<CUCV_32F> A_calc1_h(N,N,nCh);
            cuCV::Mat<CUCV_32F> A_calc2_h(N,N,nCh);
            cuCV::Mat<CUCV_32F> A_calc3_h(N,N,nCh);
            A_out_target.downloadTo(A_target_h);
            A_out_calc1.downloadTo(A_calc1_h);
            A_out_calc2.downloadTo(A_calc2_h);
            A_out_calc3.downloadTo(A_calc3_h);

            for (int ch = 0; ch < nCh; ch++) {
                for (int row = 0; row < N; row++) {
                    for (int col = 0; col < N; col++) {
                        std::stringstream rowCol; rowCol << "CUCV_32F - " << "R: "<< row <<" C: " << col << " - kSize: " << kSize;
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc1_h.at(row, col, ch), 0.0001);
                        if (kSize / 2 < 16) 
                            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc2_h.at(row, col, ch), 0.0001);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc3_h.at(row, col, ch), 0.0001);
                    }
                }
            }
        }
        {  // for CUCV_64F
            cuCV::CuMat<CUCV_64F> A = cuCV::gaussOnDevice<CUCV_64F>(N,N,nCh,sigma,true) * f;
            cuCV::CuMat<CUCV_64F> A_out_target = cuCV::zerosOnDevice<CUCV_64F>(N, N, nCh);
            cuCV::CuMat<CUCV_64F> A_out_calc1 = cuCV::zerosOnDevice<CUCV_64F>(N, N, nCh);
            cuCV::CuMat<CUCV_64F> A_out_calc2 = cuCV::zerosOnDevice<CUCV_64F>(N, N, nCh);
            cuCV::CuMat<CUCV_64F> A_out_calc3 = cuCV::zerosOnDevice<CUCV_64F>(N, N, nCh);
            cuCV::CuMat kernel = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, kSize);
            cuCV::CuMat<CUCV_32F> k_row = cuCV::createKernel(cuCV::Kernel::GAUSS, kSize, 1);
            cuCV::CuMat<CUCV_32F> k_col = cuCV::createKernel(cuCV::Kernel::GAUSS, 1, kSize);
            cuCV::simpleConv2d(A_out_target, A, kernel, cuCV::Padding::ZERO);
            cuCV::simpleSharedConv2d(A_out_calc1, A, kernel, cuCV::Padding::ZERO);
            if (kSize / 2 < 16) 
                cuCV::sharedPaddingConv2d(A_out_calc2, A, kernel, cuCV::Padding::ZERO);
            cuCV::sepSharedConv2d(A_out_calc3, A, k_row, k_col, cuCV::Padding::ZERO);
            cuCV::Mat<CUCV_64F> A_target_h(N,N,nCh);
            cuCV::Mat<CUCV_64F> A_calc1_h(N,N,nCh);
            cuCV::Mat<CUCV_64F> A_calc2_h(N,N,nCh);
            cuCV::Mat<CUCV_64F> A_calc3_h(N,N,nCh);
            A_out_target.downloadTo(A_target_h);
            A_out_calc1.downloadTo(A_calc1_h);
            A_out_calc2.downloadTo(A_calc2_h);
            A_out_calc3.downloadTo(A_calc3_h);

            for (int ch = 0; ch < nCh; ch++) {
                for (int row = 0; row < N; row++) {
                    for (int col = 0; col < N; col++) {
                        std::stringstream rowCol; rowCol << "CUCV_64F - " << "R: "<< row <<" C: " << col << " - kSize: " << kSize;
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc1_h.at(row, col, ch), 0.0001);
                        if (kSize / 2 < 16) 
                            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc2_h.at(row, col, ch), 0.0001);
                        CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(rowCol.str(), A_target_h.at(row, col, ch), A_calc3_h.at(row, col, ch), 0.0001);
                    }
                }
            }
        }
    }
}
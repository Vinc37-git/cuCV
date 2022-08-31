/**
 * @file matTest.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MATTEST_H
#define MATTEST_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include "mat.h"

#include <cppunit/extensions/HelperMacros.h>

//template <typename T>
class MatTest : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE( MatTest);
    CPPUNIT_TEST(testZeros);
    CPPUNIT_TEST(testOnes);
    CPPUNIT_TEST(testEye);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testMul);
    CPPUNIT_TEST(testDif);
    CPPUNIT_TEST(testDiv);
    CPPUNIT_TEST(testConvert);
    CPPUNIT_TEST_SUITE_END();

    protected:
    int N;
    cuCV::Mat<CUCV_8U> A8U, B8U, C8U;
    cuCV::Mat<CUCV_16U> A16U, B16U, C16U;
    cuCV::Mat<CUCV_64F> A64F, B64F, C64F;

    public:
    void setUp();
    //void tearDown();

    void testZeros();
    void testOnes();
    void testEye();
    void testAdd();
    void testMul();
    void testDif();
    void testDiv();
    void testConvert();
    //void testMatMul();
};

#endif  // MATTEST_H
/**
 * @file linalgTest.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef LINALGTEST_H
#define LINALGTEST_H
#include <iostream>
#include <stdlib.h>

#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "errorhandling.h"
#include "linalg.h"

#include <cppunit/extensions/HelperMacros.h>

class LinalgTest : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(LinalgTest);
    CPPUNIT_TEST(testNaiveMatmul);
    CPPUNIT_TEST(testMatmul);
    CPPUNIT_TEST(testSimpleConv2d);
    CPPUNIT_TEST(testSepConv2d);
    CPPUNIT_TEST(testAllConv2d);
    CPPUNIT_TEST_SUITE_END();

    protected:
    int N, nCh;

    public:
    void setUp();
    void tearDown();

    void testNaiveMatmul();
    void testMatmul();
    void testSimpleConv2d();
    void testSepConv2d();
    void testAllConv2d();
};

#endif  // LINALGTEST_H
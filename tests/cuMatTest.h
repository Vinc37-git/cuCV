/**
 * @file cuMatTest.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef CUMATTEST_H
#define CUMATTEST_H

#include "mat.h"
#include "cumat.h"
#include "initializers.h"
#include "errorhandling.h"

#include <cppunit/extensions/HelperMacros.h>

class CuMatTest : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(CuMatTest);
    CPPUNIT_TEST(testUpload);
    CPPUNIT_TEST(testUpDownload);
    CPPUNIT_TEST(testAssignment);
    CPPUNIT_TEST(testDimMismatch);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testAdd);
    CPPUNIT_TEST(testDifMulDiv);
    CPPUNIT_TEST_SUITE_END();

    protected:
    int N, nCh;

    public:
    void setUp();
    //void tearDown();

    void testUpload();
    void testUpDownload();
    void testAssignment();
    void testDimMismatch();
    void testEmpty();
    void testAdd();
    void testDifMulDiv();
};

#endif  // CUMATTEST_H
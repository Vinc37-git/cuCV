/**
 * @file cumatinitializersTest.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-08-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef CUMATINITIALIZERSTEST_H
#define CUMATINITIALIZERSTEST_H

#include "initializers.h"
#include "cumat.h"
#include "cumatinitializers.h"

#include <cppunit/extensions/HelperMacros.h>

class CuMatInitializersTest : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(CuMatInitializersTest);
    CPPUNIT_TEST(testZeros);
    CPPUNIT_TEST(testOnes);
    CPPUNIT_TEST(testEye);
    CPPUNIT_TEST(testGauss);
    CPPUNIT_TEST_SUITE_END();

    protected:
    int N;

    public:
    void setUp();
    //void tearDown();

    void testZeros();
    void testOnes();
    void testEye();
    void testGauss();
};

#endif  // CUMATINITIALIZERSTEST_H
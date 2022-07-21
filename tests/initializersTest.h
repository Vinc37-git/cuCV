/**
 * @file initializersTest.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-17
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef INITIALIZERSTEST_H
#define INITIALIZERSTEST_H

#include "mat.h"
#include "initializers.h"

#include <cppunit/extensions/HelperMacros.h>

class InitializersTest : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(InitializersTest);
    CPPUNIT_TEST(testZeros);
    CPPUNIT_TEST(testOnes);
    CPPUNIT_TEST(testEye);
    CPPUNIT_TEST_SUITE_END();

    protected:
    int N;

    public:
    void setUp();
    //void tearDown();

    void testZeros();
    void testOnes();
    void testEye();
};

#endif  // INITIALIZERSTEST_H
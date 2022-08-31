cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_executable(matConstructorsVisualTest matConstructorsVisualTest.cpp)
target_link_libraries(matConstructorsVisualTest PRIVATE cuCV_LIB)

# Testing Executables
add_executable(matTests unitTestMain.cpp matTest.cpp)
add_executable(initializersTests unitTestMain.cpp initializersTest.cpp)
add_executable(cuMatTests unitTestMain.cpp cuMatTest.cpp)
add_executable(cuMatInitializersTests unitTestMain.cpp cumatinitializersTest.cpp)
add_executable(linalgTests unitTestMain.cpp linalgTest.cpp)
#add_executable(cudeviceTests unitTestMain.cpp cudeviceTest.cpp)

# Linking
target_link_libraries(matTests PRIVATE cuCV_LIB cppunit)
target_link_libraries(initializersTests PRIVATE cuCV_LIB cppunit)
target_link_libraries(cuMatTests PRIVATE cuCV_LIB cppunit)
target_link_libraries(cuMatInitializersTests PRIVATE cuCV_LIB cppunit)
target_link_libraries(linalgTests PRIVATE cuCV_LIB cppunit)
#target_link_libraries(cudeviceTests PRIVATE cuCV_LIB cppunit)

# To make test which call CUDA codes cuda-memcheckable, define CUDA_MEMCHECK_ABLE
target_compile_definitions(cuMatTests PRIVATE CUDA_MEMCHECK_ABLE=1)
target_compile_definitions(cuMatInitializersTests PRIVATE CUDA_MEMCHECK_ABLE=1)
target_compile_definitions(linalgTests PRIVATE CUDA_MEMCHECK_ABLE=1)

add_test("CppUnitMatTests" matTests)
add_test("CppUnitInitializersTest" initializersTests)
add_test("CppUnitCuMatTests" cuMatTests)
add_test("CppUnitCuMatInitializersTests" cuMatInitializersTests)
add_test("CppUnitLinalgTests" linalgTests)
#add_test("CppUnitCudeviceTests" cudeviceTests)

# from https://stackoverflow.com/questions/17891473/cppunit-tests-always-passing-with-ctest
# "I suspect your executables are always returning 0, even if the test fails. CTest takes a return of 0 to indicate success."
#set_tests_properties(
#    "CppUnitTests"
#    PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED;Pass")



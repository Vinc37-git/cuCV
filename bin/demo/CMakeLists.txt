cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_executable(cuCVdemoArithmetic demo_arithmetic.cpp)
target_link_libraries(cuCVdemoArithmetic PRIVATE cuCV_LIB)

add_executable(cuCVdemoCopyctor demo_copyctor.cpp)
target_link_libraries(cuCVdemoCopyctor PRIVATE cuCV_LIB)

add_executable(cuCVdemoCopyMove demo_copymove.cpp)
target_link_libraries(cuCVdemoCopyMove PRIVATE cuCV_LIB)

add_executable(cudaleak cudaleak.cu)
target_link_libraries(cudaleak PRIVATE cuCV_LIB)
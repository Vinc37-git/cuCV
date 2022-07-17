cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuCV LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 52)
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

include(CTest)

add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(tests)


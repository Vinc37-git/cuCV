cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuCV LANGUAGES CXX CUDA)

add_subdirectory(src)
add_subdirectory(bin)
add_subdirectory(test)


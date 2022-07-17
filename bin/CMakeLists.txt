cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_executable(cuCV_test main.cpp)

target_link_libraries(cuCV_test PRIVATE cuCV_LIB)


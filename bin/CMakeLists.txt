cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(CUDA REQUIRED)


if(BUILD_TESTING)

  add_executable(cuCV_test main.cpp)

  set_target_properties(cuCV_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(cuCV_test PRIVATE cuCV_LIB)

endif()
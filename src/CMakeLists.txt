cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include(CTest)

add_library(particles STATIC
  randomize.cpp
  randomize.h
  particle.cu
  particle.h
  v3.cu
  v3.h
  )

  # Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(particles PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_include_directories( particles
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}) 
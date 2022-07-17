cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Find public libraries
#find_package(CUDAToolkit REQUIRED)

add_library(cuCV_LIB STATIC
        mat.cpp
        cumat.cu
        kernel.cu
        errorhandling.cpp
        initializers.cpp
        )


# Request that cuCV_LIB be built with -std=c++11
# As this is a public compile feature anything that links to cuCV_LIB
# will also build with -std=c++11
target_compile_features(cuCV_LIB PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the cuCV_LIB
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(cuCV_LIB
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )


target_include_directories(cuCV_LIB
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}) 
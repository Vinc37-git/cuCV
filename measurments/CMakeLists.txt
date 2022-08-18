cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Build executables for runtime measurments.
# 3 executables will be build for one cuCV datatype each.

add_executable(cucvRuntimeMeasurments8U cucvRuntimeMeasurments.cpp)
target_compile_definitions(cucvRuntimeMeasurments8U PRIVATE CUCV_XX=CUCV_8U)
target_link_libraries(cucvRuntimeMeasurments8U PRIVATE cuCV_LIB ${OpenCV_LIBS})

add_executable(cucvRuntimeMeasurments16U cucvRuntimeMeasurments.cpp)
target_compile_definitions(cucvRuntimeMeasurments16U PRIVATE CUCV_XX=CUCV_16U)
target_link_libraries(cucvRuntimeMeasurments16U PRIVATE cuCV_LIB ${OpenCV_LIBS})

add_executable(cucvRuntimeMeasurments64F cucvRuntimeMeasurments.cpp)
target_compile_definitions(cucvRuntimeMeasurments64F PRIVATE CUCV_XX=CUCV_64F)
target_link_libraries(cucvRuntimeMeasurments64F PRIVATE cuCV_LIB ${OpenCV_LIBS})
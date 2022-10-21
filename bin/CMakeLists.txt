cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_subdirectory(demo)

#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cucvconv cucvconv.cpp)
target_link_libraries(cucvconv PRIVATE cuCV_LIB ${OpenCV_LIBS})


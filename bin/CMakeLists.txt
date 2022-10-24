cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_subdirectory(demo)

#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cucvconv cucvconv.cpp)
target_link_libraries(cucvconv PRIVATE cuCV_LIB)

add_executable(addition addition.cpp)
target_link_libraries(addition PRIVATE cuCV_LIB)

add_executable(matmul matmul.cpp)
target_link_libraries(matmul PRIVATE cuCV_LIB)

add_executable(conv conv.cpp)
target_link_libraries(conv PRIVATE cuCV_LIB)

add_executable(memaccess memaccess.cu)
target_link_libraries(memaccess PRIVATE cuCV_LIB)


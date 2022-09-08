/**
 * @file errorhandling.cpp
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "errorhandling.h"


void cuCV::error::gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s (FILE: %s), (LINE: %d)\n", cudaGetErrorString(code), file, line);
      if (abort) 
        exit(code);
   }
}


int cuCV::error::cudaError(std::string pos, cudaError_t & err) {
    std::string msg;
    if ((int) err != 0) {
        msg = "CUDA Error Code: " + std::to_string((int) err);
        printf("%s\n", msg.c_str());
        throw cudaGetErrorName(err);
    }

    return err == 0;
}



template <typename T>
cuCV::exception::DimensionMismatch<T>::DimensionMismatch(const Mat<T> & A, const Mat<T> & B, std::string desc) 
        : aX(A.getWidth()), aY(A.getHeight()), aZ(A.getNChannels()), bX(B.getWidth()), bY(B.getHeight()), bZ(B.getNChannels()), operation(desc) {
    genMessage();
}

template <typename T>
cuCV::exception::DimensionMismatch<T>::DimensionMismatch(const CuMat<T> & A, const CuMat<T> & B, std::string desc) 
        : aX(A.getWidth()), aY(A.getHeight()), aZ(A.getNChannels()), bX(B.getWidth()), bY(B.getHeight()), bZ(B.getNChannels()), operation(desc) {
    genMessage();
}

template <typename T>
cuCV::exception::DimensionMismatch<T>::DimensionMismatch(const CuMat<T> & A, const Mat<T> & B, std::string desc) 
        : aX(A.getWidth()), aY(A.getHeight()), aZ(A.getNChannels()), bX(B.getWidth()), bY(B.getHeight()), bZ(B.getNChannels()), operation(desc) {
    genMessage();
}

template <typename T>
cuCV::exception::DimensionMismatch<T>::DimensionMismatch(const Mat<T> & A, const CuMat<T> & B, std::string desc) 
        : aX(A.getWidth()), aY(A.getHeight()), aZ(A.getNChannels()), bX(B.getWidth()), bY(B.getHeight()), bZ(B.getNChannels()), operation(desc) {
    genMessage();
}

template <typename T>
void cuCV::exception::DimensionMismatch<T>::genMessage() {
    msg = std::string("Dimension Error! You are trying to perform an ") + operation + " "
        + std::string("but dimensions do not match. \nDimensions are: A: (")
        + std::to_string(aX) + ", " + std::to_string(aY) + ", " + std::to_string(aZ) + ") vs B: ("
        + std::to_string(bX) + ", " + std::to_string(aY) + ", " + std::to_string(aZ) + ").\n";  
}

template <typename T>
const char * cuCV::exception::DimensionMismatch<T>::what() const throw() {
    return msg.c_str();
}


cuCV::exception::NullPointer::NullPointer(const char * msg) 
        : msg(msg) { }

const char * cuCV::exception::NullPointer::what() const throw() {
        return msg;
}


template class cuCV::exception::DimensionMismatch<CUCV_8U>;
template class cuCV::exception::DimensionMismatch<CUCV_16U>;
template class cuCV::exception::DimensionMismatch<CUCV_32F>;
template class cuCV::exception::DimensionMismatch<CUCV_64F>;

/**
 * @file errorhandling.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-31
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef ERRORHANDLING_H
#define ERRORHANDLING_H

#include <iostream>
#include <exception>
#include <cuda_runtime.h>

#include "mat.h"
#include "cumat.h"


/**
 * @brief A macro to wrap all cuda runtime api calls. 
 * 
 */
#define gpuErrchk(ans) { cuCV::error::gpuAssert((ans), __FILE__, __LINE__); }


namespace cuCV {

template <typename T>
class CuMat;  // Forward Declaration of CuMat to make sure compiler knows the class exists


/**
 * @brief A collection of error handling functions. 
 */
namespace error {

/**
 * @brief Canoncial Way to check for errors by cuda runtime api.
 * from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * 
 * @param code 
 * @param file 
 * @param line 
 * @param abort 
 */
void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true);


/**
 * @brief DEPRACTED - use gpuAssert instead - A function to translate the cuda error code.
 * 
 * @param pos 
 * @param err 
 * @return int 
 */
int cudaError(std::string pos, cudaError_t & err);

};  // namespace error


/**
 * @brief A collection of custom cuCV exceptions. 
 */
namespace exception {


/**
 * @brief Dimension Mismatch Exception. 
 * A custom exception which will be thrown when the dimensions of two matrices 
 * do not match for an operation or data up- / download. The matrices may be of type Mat or CuMat.
 * 
 * @tparam T 
 */
template <typename T>
class DimensionMismatch : public std::exception {
public:
    DimensionMismatch();
    DimensionMismatch(const cuCV::Mat<T> & A, const cuCV::Mat<T> & B, std::string desc="operation");
    DimensionMismatch(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B, std::string desc="operation");
    DimensionMismatch(const cuCV::CuMat<T> & A, const cuCV::Mat<T> & B, std::string desc="download");
    DimensionMismatch(const cuCV::Mat<T> & A, const cuCV::CuMat<T> & B, std::string desc="upload");

    const char * what() const throw();
    void genMessage();

private:
    std::string msg;
    std::string operation;
    const int aX, aY, aZ, bX, bY, bZ;
};


/**
 * @brief Nullpointer exception. Is thrown when an operand points to NULL.
 */
class NullPointer : public std::exception {
public:
    NullPointer(const char * msg);

    const char * what() const throw();

    const char * msg;
};

};  // namespace exception

}

#endif //
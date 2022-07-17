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
#include <cuda_runtime.h>

#define gpuErrchk(ans) { cuCV::error::gpuAssert((ans), __FILE__, __LINE__); }



namespace cuCV {
namespace error {

/**
 * @brief Canoncial Way to check for errors
 * from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * 
 * @param code 
 * @param file 
 * @param line 
 * @param abort 
 */
void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true);


int cudaError(std::string pos, cudaError_t & err);

}
}


#endif //
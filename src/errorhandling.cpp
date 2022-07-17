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



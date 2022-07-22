/**
 * @file linalg.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef LINALG_H
#define LINALG_H

#include <iostream>
#include <unistd.h>

#include "errorhandling.h"
#include "mat.h"
#include "kernel.h"
#include "kernelcumat.h"


namespace cuCV {


template <typename T>
CuMat<T> naiveMatmul(const CuMat<T> & A, const CuMat<T> & B);


template <typename T>
CuMat<T> matmul(const CuMat<T> & A, const CuMat<T> & B);


}  // namepsace cuCV

#endif  // LINALG_H
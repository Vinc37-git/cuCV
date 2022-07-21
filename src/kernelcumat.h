/**
 * @file kernelcumat.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef KERNELCUMAT_H
#define KERNELCUMAT_H

#include "cumat.h"

namespace cuCV {


template <typename T>
class CuMat;  ///< Forward Declaration of CuMat to make sure compiler knows the class exists

/**
 * @brief Kernel CuMat Class is a sibling of the standard CuMat Class. However, it only borrows a reference to the data
 * of an cuMat Object to launch the cuda kernel. This is neccessary due to the fact, that __global__ cuda kernels require 
 * arguments to be passed by value, which would result in a copy of data when using the standard cuMat object.
 * Since the data is borrowed only, the KernelCuMat object will not take care of the data. When the object goes out of scope
 * after the kernel call, it will not free the data associated with the borrowed reference.
 * 
 * @tparam T 
 */
template <typename T>
class KernelCuMat : public Mat<T> {
public:
    /**
     * @brief Construct a new Kernel Cu Mat object
     */
    KernelCuMat();


    /**
     * @brief Construct a new Kernel Cu Mat object using a cuMat object.
     * 
     * @param cuMat 
     */
    KernelCuMat(const CuMat<T> & cuMat);


    /**
     * @brief Construct a new Kernel Cu Mat object by copying another kernelCuMat object.
     * 
     * @param kernelCuMat 
     */
    KernelCuMat(const KernelCuMat<T> & kernelCuMat);


    /**
     * @brief Destroy the Kernel Cu Mat object. No data will be freed.
     */
    ~KernelCuMat();
};
};

#endif  // KERNELCUMAT_H
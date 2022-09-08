/**
 * @file kernel.h
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef KERNEL_H
#define KERNEL_H

#include <cmath>
#include <cuda_runtime.h>

#include "mat.h"
#include "cumat.h"
#include "filter.h"

#define CUCV_PRINT_THREAD(str, ...) \
        printf("(%d, %d, %d) : " str, threadIdx.x, threadIdx.y, threadIdx.z , __VA_ARGS__);


/// @note: never tested.
#define CUCV_PRINT_SHARED(sharedMem) {\
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) { \
                for (int i = 0; i<blockDim.x * blockDim.y * 9; ++i) { \
                        printf("%.1f ", (double) sharedMem[i]); \
                        if ((i+1) % (blockDim.x * 3) == 0) \
                                printf("\n"); \
                } \
                printf("\n"); \
        } \
}


namespace cuCV {


template <typename T>
class CuMat;  // Forward Declaration of CuMat to make sure compiler knows the class exists

template <typename T>
class DeviceCuMat;  // Forward Declaration of CuMat to make sure compiler knows the class exists

enum class Padding;


/**
 * @brief A collection of CUDA kernels (__global__ functions). Those functions are called from the host (CPU)
 * to launch calculations on the device (GPU) unless you want to use CUDA dynamic parallelism. 
 * Arguments in CUDA kernels must be passed by value and can not be passed
 * by reference. Hence, kernel arguments should not cuMat object since this would lead to the copy of large data chunks.
 * Rather use the class DeviceCuMat, which is a simple copy of CuMat class with the difference that data is only borrowed 
 * but not copied.
 * 
 * 
 */
namespace kernel {


/**
 * @brief CUDA kernel to perform an elementwise addition on the device of two Matrices.
 * Note that shape of A must be equal to shape of B and Out.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T> __global__ 
void add(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform an elementwise addition on the device of one Matrix with a scalar.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T> __global__ 
void add(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


/**
 * @brief CUDA kernel to perform a elementwise subtraction on the device of two Matrices.
 * Note that shape of A must be equal to shape of B and Out.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T> __global__ 
void dif(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform a elementwise subtraction on the device of two Matrices.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T> __global__ 
void dif(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


/**
 * @brief CUDA kernel to perform a elementwise multiplication on the device of two Matrices.
 * Note that shape of A must be equal to shape of B and Out.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T> __global__ 
void mul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform a elementwise multiplication on the device of one Matrix with a scalar.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T> __global__ 
void mul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


/**
 * @brief CUDA kernel to perform a elementwise division on the device of two Matrices.
 * Note that shape of A must be equal to shape of B and Out.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param B The second operand.
 */
template <typename T> __global__ 
void div(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform a elementwise division on the device of one Matrix with a scalar.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param alpha The second operand.
 */
template <typename T> __global__ 
void div(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const T alpha);


/**
 * @brief CUDA kernel to perform a elementwise division on the device of one Matrix with a floating point sclar.
 * @note: pAlpha is a pointer to a floating point value on the device.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results.
 * @param A The first operand.
 * @param pAlpha The second operand of type float on the device (pointer to float).
 */
template <typename T> __global__ 
void div(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, float * pAlpha);


/**
 * @brief CUDA kernel to perform a matrix multiplication on the device of two matrices 
 * without using the concept of shared memory. 
 * @todo rename to simpleMatmul
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results with shape NxL.
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 */
template <typename T> __global__ 
void naiveMatmul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform a matrix multiplication on the device of two matrices 
 * using the concept of shared memory.
 * @todo Explain a bit about shared memory. 
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the results with shape NxL.
 * @param A The first operand with shape NxM.
 * @param B The second operand with shape MxL.
 */
template <typename T> __global__ 
void matmul(cuCV::DeviceCuMat<T> OUT, const cuCV::DeviceCuMat<T> A, const cuCV::DeviceCuMat<T> B);


/**
 * @brief CUDA kernel to perform a convolution on `A` with the kernel kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method.
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @param OUT The output matrix which will keep the results of the convolution. The size dependes on the Padding method.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 */
template <typename T1, typename T2> __global__
void simpleConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding);


/**
 * @brief CUDA kernel to perform a convolution on `A` with the kernel kernel. `A` can be padded according to Padding enum.
 * The calculation is performed by the simplest convolution method using shared memory.
 * Both the image (as tiles) and the kernel is loaded into shared memory. The kernel memory is allocated dynamically to match
 * the current kernel size. Therefore, the kernel size must be passed as third parameter in the kernel launch execution parameter.
 * @note that during computation, values which extend outwards of one block are loaded from global memory as they are not existent
 * in the current block's shared memory.
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F. Use CUCV_32F for optimal performance.
 * @param OUT The output matrix which will keep the results of the convolution. The size dependes on the Padding method.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param padding The padding method.
 */
template <typename T1, typename T2> __global__
void simpleSharedConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding);


template <typename T1, typename T2> __global__
void simpleSharedConv2d_2(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding);


template <typename T1, typename T2> __global__
void sepColConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding);


/**
 * @brief CUDA kernel to initialize a matrix on the device. The matrix is filled with 0s.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the 0s.
 */
template <typename T> __global__ 
void zeros(cuCV::DeviceCuMat<T> OUT);


/**
 * @brief CUDA kernel to initialize a matrix on the device. The matrix is filled with 1s.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the 1s.
 */
template <typename T> __global__ 
void ones(cuCV::DeviceCuMat<T> OUT);


/**
 * @brief CUDA kernel to initialize a matrix on the device. 
 * The matrix is filled with ones on the diagonal and zeros elsewhere.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the result.
 */
template <typename T> __global__ 
void eye(cuCV::DeviceCuMat<T> OUT);


/**
 * @brief CUDA kernel to initialize a matrix on the device. 
 * The matrix is filled with elements which entries follow a gaussian distribution.
 * 
 * @tparam T CUCV datatype: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F
 * @param OUT The output matrix which will keep the result.
 */
template <typename T> __global__ 
void gauss(cuCV::DeviceCuMat<T> OUT, double sigma, bool norm, float * sum);


};  // namespace kernel
};  // namespace cuCV


#endif //
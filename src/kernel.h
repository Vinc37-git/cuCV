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
#define CUCV_PRINT_SHARED(sharedMem, blocksX, blocksY) {\
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0) { \
                for (int i = 0; i<blockDim.x * blockDim.y * blocksX * blocksY; ++i) { \
                        printf("%.1f ", (double) sharedMem[i]); \
                        if ((i+1) % (blockDim.x * blocksX) == 0) \
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
 * by reference. Hence, kernel arguments should not be cuMat object since this would lead to the copy of large data chunks.
 * Rather use the class DeviceCuMat, which is a simple copy of CuMat class and its attributes (including pointers)
 * with the difference that data is only borrowed but not copied.
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
 * The calculation is performed by the simplest convolution method. Every element required during the computation is
 * loaded from global memory.
 * @todo complexity
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
 * Note that during computation, values which extend outwards of one block are loaded from global memory as they are not existent
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


/**
 * @brief CUDA kernel to perform a convolution on `A` with the kernel kernel. `A` can be padded according to Padding enum.
 * The function takes advantage of shared memory. The function can be split into two parts:
 * the loading step and the calculation step. During the loading step, every device block loads a subtile of A into shared memory 
 * which size is equal to the block dimension. Additionally, the tile is padded with 8 other shared memory tiles, where the neighbouring
 * pixels are loaded into. As a result, the shared memory is of size `blockDim.x * blockDim.y * 9`, what makes this methods limitation
 * obvious: the shared memory size per multiprocessor. Hence, this convolution function is limited to kernel sizes len(side) < max(blockDim) * 2. 
 * However, it is very fast for kernel sizes which are shortly below the limitation since all apron values are loaded into shared
 * memory before calculation. However, it is NOT recommended for small kernel sizes relative to the block dimensions,
 * since in that case a lot of neighbouring pixels are loaded into shared memory, which are acutally not neccessary for calculation.
 * @todo: @doc: Complexity
 * @todo A dynamic approach to control the shared memory padding size.
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F. Use CUCV_32F for optimal performance.
 * @param OUT The output matrix which will keep the results of the convolution. The size dependes on the Padding method.
 * @param A The Matrix to be convoluted.
 * @param kernel The kernel for the convolution. Note that the kernel will be flipped (since it is a convolution and not correlation).
 * @param shElemsA The number of elements of A in the shared memory array.
 * @param padding The padding method.
 */
template <typename T1, typename T2> __global__
void sharedPaddingConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const size_t shElemsA, const cuCV::Padding padding);


/**
 * @brief CUDA kernel to perform a convolution on `A` with the kernel kernel. `A` can be padded according to Padding enum.
 * This function performs a 1d convolution with a row only kernel on a (linearly stored) 2d array. Hence, the convolution
 * is performed in column / horizontal direction. It takes advantage of shared memory. The function is basically split
 * into two parts: the loading and the calculation step. During the loading step, A is split into row vectors / tiles of
 * length X. Those tiles are padded with pixels of neighbouring columns dependent on the filter size and loaded into shared memory.
 * In contrast to the function `sharedPaddingConv2d()`, the limiting factor is now the available threads rather than the maxmimum
 * shared memory available. In the calculation step, the row kernel is convoluted over the tile of A and results are written into OUT.
 * This CUDA kernel can be combined with sepColConv2d() to perform a full 2d convolution on images with linearly seperable kernels/filters.
 * To increase the performance during the loading stage, the alignment of threads and warps is considered: reading of global memory is fastest, when
 * memory transactions are coalesced. This is, when all threads within one warp access consecutive words in memory. Then, the access of all
 * threads within the warp is done in one memory transcation. Additionally, th base read/write addresses of the first thread of a warp must meet half-warp
 * alignment requirement. To archive that, a alignment offset is passed to the thread size configuration.
 * @todo: @doc: complexity 
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F. Use CUCV_32F for optimal performance.
 * @param OUT The output matrix which will keep the results of the convolution. The size dependes on the Padding method.
 * @param A The Matrix to be convoluted.
 * @param rowKernel The row (horizontal) kernel. Note that if the kernel's heigth > 1, only the first row will be used for the convolution.
 * @param tileWidth The width of the tile / row vector of A.
 * @param shElemsA The number of elements of A in the shared memory array.
 * @param padding The padding method.
 */
template <typename T1, typename T2> __global__
void sepRowConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);


/**
 * @brief CUDA kernel to perform a convolution on `A` with the kernel kernel. `A` can be padded according to Padding enum.
 * This function performs a 1d convolution with a column only kernel on a (linearly stored) 2d array. Hence, the convolution 
 * is performed in row / vertical direction. It takes advantage of shared memory. The function is basically split
 * into two parts: the loading and the calculation step. During the loading step, A is split into tiles of 32 x Y. Those tiles
 * are padded with pixels of neighbouring rows dependent on the filter size and loaded into shared memory. The width of the tiles
 * is 32 to make sure that one warp access one row of the tile coalesced for every available CUCV datatype. Note that this is only
 * valid for every warp, if the image size is a multiple of warp size. Equivalent to sepRowConv2d(),
 * the limiting factor is the available threads rather than the maxmimum shared memory available 
 * (in constrast to function `sharedPaddingConv2d()`). In the calculation step, the column kernel is convoluted over the tile of A
 * and results are written into OUT. This CUDA kernel can be combined with sepColConv2d() to perform a full 2d convolution on images with
 * linearly seperable kernels/filters.
 * @todo: @doc: complexity 
 * 
 * @tparam T1 CUCV datatype of `OUT` and `A`: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F.
 * @tparam T2 CUCV datatype of the kernel: CUCV_8U, CUCV_16U, CUCV_32F, CUCV_64F. Use CUCV_32F for optimal performance.
 * @param OUT The output matrix which will keep the results of the convolution. The size dependes on the Padding method.
 * @param A The Matrix to be convoluted.
 * @param colKernel The width (vertical) kernel. Note that if the kernel's width > 1, only the first column will be used for the convolution.
 * @param tileWidth The width of the tile of A.
 * @param tileHeight The height of the tile of A.
 * @param shElemsA The number of elements of A in the shared memory array.
 * @param padding The padding method.
 */
template <typename T1, typename T2> __global__
void sepColConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> colKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);


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
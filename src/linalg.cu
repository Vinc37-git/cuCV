/**
 * @file linalg.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-07-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "linalg.h"


template <typename T>
cuCV::CuMat<T> cuCV::naiveMatmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.mWidth != B.mHeight || A.mChannels != B.mChannels)
        throw cuCV::exception::DimensionMismatch(A, B, "naive matmul operation");
    
    if (A.mData == NULL || B.mData == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    cuCV::CuMat<T> out(A.mHeight, B.mWidth, A.mChannels);
    out.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((out.mWidth + threads.x - 1) / threads.x, (out.mHeight + threads.y - 1) / threads.y, out.mChannels);

    /// Perform Math
    cuCV::kernel::naiveMatmul<<<blocks, threads>>>(out.kernel(), A.kernel(), B.kernel());

    return out;
}


template <typename T>
cuCV::CuMat<T> cuCV::matmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.mWidth != B.mHeight || A.mChannels != B.mChannels)
        throw cuCV::exception::DimensionMismatch(A, B, "matmul operation");
    
    if (A.mData == NULL || B.mData == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    cuCV::CuMat<T> out(A.mHeight, B.mWidth, A.mChannels);
    out.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((out.mWidth + threads.x - 1) / threads.x, (out.mHeight + threads.y - 1) / threads.y, out.mChannels);

    /// Perform Math
    cuCV::kernel::matmul<<<blocks, threads>>>(out.kernel(), A.kernel(), B.kernel());

    return out;
}


/// Explicit template specialization
template cuCV::CuMat<CUCV_8U> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template cuCV::CuMat<CUCV_16U> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template cuCV::CuMat<CUCV_64F> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template cuCV::CuMat<CUCV_8U> cuCV::matmul(const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template cuCV::CuMat<CUCV_16U> cuCV::matmul(const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template cuCV::CuMat<CUCV_64F> cuCV::matmul(const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

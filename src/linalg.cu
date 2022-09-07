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
void cuCV::simpleMatmul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.mWidth != B.mHeight || A.mChannels != B.mChannels)
        throw cuCV::exception::DimensionMismatch(A, B, "matmul operation");

    if (OUT.mHeight != A.mHeight || OUT.mWidth != B.mWidth)
        throw std::runtime_error("DimensionMismatch: Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l).");
    
    if (A.mData == NULL || B.mData == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y, OUT.mChannels);

    /// Perform Math
    cuCV::kernel::naiveMatmul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
cuCV::CuMat<T> cuCV::naiveMatmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    cuCV::CuMat<T> OUT(A.mHeight, B.mWidth, A.mChannels);
    OUT.allocateOnDevice();

    cuCV::simpleMatmul(OUT, A, B);

    return OUT;
}


template <typename T>
void cuCV::matmul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.mWidth != B.mHeight || A.mChannels != B.mChannels)
        throw cuCV::exception::DimensionMismatch(A, B, "matmul operation");

    if (OUT.mHeight != A.mHeight || OUT.mWidth != B.mWidth)
        throw std::runtime_error("DimensionMismatch: Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l).");
    
    if (A.mData == NULL || B.mData == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y, OUT.mChannels);

    /// Perform Math
    cuCV::kernel::matmul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
cuCV::CuMat<T> cuCV::matmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {

    cuCV::CuMat<T> OUT(A.mHeight, B.mWidth, A.mChannels);
    OUT.allocateOnDevice();

    cuCV::matmul(OUT, A, B);

    return OUT;
}


template <typename T1, typename T2> 
void cuCV::simpleConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    if (A.getDataPtr() == NULL || kernel.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (A.getWidth() < kernel.getWidth() || A.getHeight() < kernel.getHeight()) {
        std::string msg = "DimensionError: kernel size is larger than matrix size: (" 
                + std::to_string(A.getWidth()) + ", " + std::to_string(A.getHeight()) + ") vs ("
                + std::to_string(kernel.getWidth()) + ", " + std::to_string(kernel.getHeight()) + ")";
        throw std::runtime_error(msg);
    }

    // Determine size of output matrix
    int outW = 0, outH = 0;
    switch (padding) {
        case cuCV::Padding::ZERO:
            outW = A.getWidth(), outH = A.getHeight();
            break;
        default:
            break;
    }

    // Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y, OUT.mChannels);

    cuCV::kernel::simpleConv2d<<<blocks, threads>>>(OUT.kernel(), A.kernel(), kernel.kernel(), padding);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T1, typename T2> 
cuCV::CuMat<T1> cuCV::simpleConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    // Determine size of output matrix
    int outW = 0, outH = 0;
    switch (padding) {
        case cuCV::Padding::ZERO:
            outW = A.getWidth(), outH = A.getHeight();
            break;
        default:
            break;
    }

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::simpleConv2d(out, A, kernel, padding);

    return out;
}


template <typename T1, typename T2> 
void cuCV::simpleSharedConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    if (A.getDataPtr() == NULL || kernel.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (A.getWidth() < kernel.getWidth() || A.getHeight() < kernel.getHeight()) {
        std::string msg = "DimensionError: kernel size is larger than matrix size: (" 
                + std::to_string(A.getWidth()) + ", " + std::to_string(A.getHeight()) + ") vs ("
                + std::to_string(kernel.getWidth()) + ", " + std::to_string(kernel.getHeight()) + ")";
        throw std::runtime_error(msg);
    }

    // Determine size of output matrix
    int outW = 0, outH = 0;
    switch (padding) {
        case cuCV::Padding::ZERO:
            outW = A.getWidth(), outH = A.getHeight();
            break;
        default:
            break;
    }

    // Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.mWidth + threads.x - 1) / threads.x, (OUT.mHeight + threads.y - 1) / threads.y, OUT.mChannels);
    const size_t kernelCounts = (size_t) (kernel.getWidth() * kernel.getHeight() * sizeof(T2));

    cuCV::kernel::simpleSharedConv2d<<<blocks, threads, kernelCounts>>>(OUT.kernel(), A.kernel(), kernel.kernel(), padding);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T1, typename T2> 
cuCV::CuMat<T1> cuCV::simpleSharedConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    // Determine size of output matrix
    int outW = 0, outH = 0;
    switch (padding) {
        case cuCV::Padding::ZERO:
            outW = A.getWidth(), outH = A.getHeight();
            break;
        default:
            break;
    }

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::simpleSharedConv2d(out, A, kernel, padding);

    return out;
}


// template <typename T>
// cuCV::CuMat<T> cuCV::simpleConv2d(const CuMat<T> & A, const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY, const cuCV::Padding padding) {
//     // Create kernel
    
// }


/// Explicit template specialization
template void cuCV::simpleMatmul(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::simpleMatmul(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::simpleMatmul(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::simpleMatmul(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template cuCV::CuMat<CUCV_8U> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template cuCV::CuMat<CUCV_16U> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template cuCV::CuMat<CUCV_32F> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template cuCV::CuMat<CUCV_64F> cuCV::naiveMatmul(const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::matmul(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::matmul(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::matmul(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::matmul(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template cuCV::CuMat<CUCV_8U> cuCV::matmul(const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template cuCV::CuMat<CUCV_16U> cuCV::matmul(const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template cuCV::CuMat<CUCV_32F> cuCV::matmul(const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template cuCV::CuMat<CUCV_64F> cuCV::matmul(const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::simpleSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

template cuCV::CuMat<CUCV_8U> cuCV::simpleSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::simpleSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::simpleSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::simpleSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::simpleSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

// template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY, const cuCV::Padding padding);
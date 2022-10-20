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
void cuCV::add(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);

    if (!A.compareDim(A, B))
        throw cuCV::exception::DimensionMismatch(A, B);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::add<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
void cuCV::add(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const T alpha) {
    if (A.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::add<<<blocks, threads>>>(OUT.kernel(), A.kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
void cuCV::dif(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);

    if (!A.compareDim(A, B))
        throw cuCV::exception::DimensionMismatch(A, B);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::dif<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
}


template <typename T>
void cuCV::dif(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const T alpha) {
    if (A.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::dif<<<blocks, threads>>>(OUT.kernel(), A.kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
void cuCV::mul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);

    if (!A.compareDim(A, B))
        throw cuCV::exception::DimensionMismatch(A, B);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::mul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
}


template <typename T>
void cuCV::mul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const T alpha) {
    if (A.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::mul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
void cuCV::div(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);

    if (!A.compareDim(A, B))
        throw cuCV::exception::DimensionMismatch(A, B);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::div<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); 
}


template <typename T>
void cuCV::div(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const T alpha) {
    if (A.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("Calculation failed. One or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateLike(A);
    
    if (!A.compareDim(OUT, A))
        throw cuCV::exception::DimensionMismatch(OUT, A);

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::div<<<blocks, threads>>>(OUT.kernel(), A.kernel(), alpha);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}



template <typename T>
void cuCV::simpleMatmul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.getWidth() != B.getHeight() || A.getNChannels() != B.getNChannels())
        throw cuCV::exception::DimensionMismatch(A, B, "matmul operation");

    if (OUT.getHeight() != A.getHeight() || OUT.getWidth() != B.getWidth())
        throw std::runtime_error("DimensionMismatch: Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l).");
    
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::naiveMatmul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
cuCV::CuMat<T> cuCV::naiveMatmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    cuCV::CuMat<T> OUT(A.getHeight(), B.getWidth(), A.getNChannels());
    OUT.allocateOnDevice();

    cuCV::simpleMatmul(OUT, A, B);

    return OUT;
}


template <typename T>
void cuCV::matmul(cuCV::CuMat<T> & OUT, const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {
    // Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l)
    if (A.getWidth() != B.getHeight() || A.getNChannels() != B.getNChannels())
        throw cuCV::exception::DimensionMismatch(A, B, "matmul operation");

    if (OUT.getHeight() != A.getHeight() || OUT.getWidth() != B.getWidth())
        throw std::runtime_error("DimensionMismatch: Matrices must have a shape that matches the signature A(m,n) @ B(n,l)->(m,l).");
    
    if (A.getDataPtr() == NULL || B.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    /// Perform Math
    cuCV::kernel::matmul<<<blocks, threads>>>(OUT.kernel(), A.kernel(), B.kernel());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T>
cuCV::CuMat<T> cuCV::matmul(const cuCV::CuMat<T> & A, const cuCV::CuMat<T> & B) {

    cuCV::CuMat<T> OUT(A.getHeight(), B.getWidth(), A.getNChannels());
    OUT.allocateOnDevice();

    cuCV::matmul(OUT, A, B);

    return OUT;
}


template <typename T1, typename T2> 
void cuCV::simpleConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    cuCV::convHelper::checks(A, kernel);

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    ///< @todo Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());

    cuCV::kernel::simpleConv2d<<<blocks, threads>>>(OUT.kernel(), A.kernel(), kernel.kernel(), padding);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T1, typename T2> 
cuCV::CuMat<T1> cuCV::simpleConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::simpleConv2d(out, A, kernel, padding);

    return out;
}


template <typename T1, typename T2> 
void cuCV::simpleSharedConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    cuCV::convHelper::checks(A, kernel);

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    /// @todo Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());
    const size_t kernelCounts = (size_t) (kernel.getWidth() * kernel.getHeight() * sizeof(T2));

    cuCV::kernel::simpleSharedConv2d<<<blocks, threads, kernelCounts>>>(OUT.kernel(), A.kernel(), kernel.kernel(), padding);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T1, typename T2> 
cuCV::CuMat<T1> cuCV::simpleSharedConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::simpleSharedConv2d(out, A, kernel, padding);

    return out;
}


template <typename T1, typename T2> 
void cuCV::sharedPaddingConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    cuCV::convHelper::checks(A, kernel);

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    /// @todo Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    // Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
    // As this function requires a lot of shared memory, a BLOCK_SIZE of 32x32 is to large in combination with 64 bit 
    // floating type matrices: sizeof(sharedMem) = 9 * BLOCK_SIZE * BLOCK_SIZE * sizeof(double)
    const size_t block_size = std::min(BLOCK_SIZE, 16);    
    const dim3 threads(block_size, block_size);
    const dim3 blocks((OUT.getWidth() + threads.x - 1) / threads.x, (OUT.getHeight() + threads.y - 1) / threads.y, OUT.getNChannels());
    const size_t shCountsA = cuCV::convHelper::cucvAddBytes2Align(block_size * block_size * 9 * sizeof(T1), sizeof(T2));
    const size_t shMemCounts = shCountsA + (size_t) (kernel.getWidth() * kernel.getHeight() * sizeof(T2));

    /// @todo: Check if block_size < kernel.getWidth() / 2
    if (block_size < std::max(kernel.getWidth() / 2,  kernel.getHeight() / 2 ))
        throw std::runtime_error("Error: sharedPaddingConv2d() not allowed for kernel side length: " 
                + std::to_string(std::max(kernel.getHeight(), kernel.getWidth())) + " / 2 > " + std::to_string(block_size));

    cuCV::kernel::sharedPaddingConv2d<<<blocks, threads, shMemCounts>>>(OUT.kernel(), A.kernel(), kernel.kernel(), shCountsA / sizeof(T1), padding);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}


template <typename T1, typename T2> 
cuCV::CuMat<T1> cuCV::sharedPaddingConv2d(const CuMat<T1> & A, const CuMat<T2> & kernel, const cuCV::Padding padding) {

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::sharedPaddingConv2d(out, A, kernel, padding);

    return out;
}


template <typename T1, typename T2>
void cuCV::sepSharedConv2d(CuMat<T1> & OUT, const CuMat<T1> & A, const CuMat<T2> & rowKernel, const CuMat<T2> & colKernel, const cuCV::Padding padding) {
    
    cuCV::convHelper::checks(A, rowKernel);

    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    /// @todo Check if OUT matches determined size

    // Check if OUT points to data
    if (OUT.getDataPtr() == NULL)
        OUT.allocateOnDevice();

    /** Call CUDA kernel: sepRowConv2d():
     * 
     * Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
     * rowKernel must be of shape (W,1) (horizontal vector)
     * blockDim should be of shape: (X,1), where X = KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS.
     * gridDim should be of shape: ((A.mWidth / ROW_TILE_WIDTH + 1), A.mHeight, A.mChannels), resulting in one block per tile.
     * Hence, it is independet of threadsIdx (in contrast to all other kernel calls in cuCV).
     * A shared memory of size (W // 2 + ROW_TILE_WIDTH + W // 2) * sizeof(T1) + rowKernel.mWidth * sizeof(T2) will be allocated dynamically.
     */
    {
        const size_t TILE_WIDTH_X = 128, kNx = (size_t) rowKernel.getWidth() / 2, ALIGNMENT_OFFSET = 0, kNx_aligned = kNx + ALIGNMENT_OFFSET;   
        const dim3 threads(kNx_aligned + TILE_WIDTH_X + kNx);
        const dim3 blocks((A.getWidth() + TILE_WIDTH_X - 1) / TILE_WIDTH_X, A.getHeight(), A.getNChannels());
        const size_t shCountsA = cuCV::convHelper::cucvAddBytes2Align((TILE_WIDTH_X + 2 * kNx) * sizeof(T1), sizeof(T2));
        const size_t shMemCounts = shCountsA + (size_t) (rowKernel.getWidth() * rowKernel.getHeight() * sizeof(T2));
                //+ (TILE_WIDTH_X + 2 * kNx) * sizeof(T1);

        /// @todo: Check if block_size < rowKernel.getWidth() / 2
        /// @todo: Warning or automatic reshape if kernel is not a row kernel
        if (rowKernel.getWidth() <= 1 && rowKernel.getHeight() > 1)
            fprintf(stderr, "Warning: Row Kernel width for seperated convolution is %d, but height is: %d.\n", rowKernel.getWidth(), rowKernel.getHeight());

        cuCV::kernel::sepRowConv2d<<<blocks, threads, shMemCounts>>>(OUT.kernel(), A.kernel(), rowKernel.kernel(), TILE_WIDTH_X, shCountsA / sizeof(T1), padding);
        
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    

    /** Call CUDA kernel: sepColConv2d():
     * 
     * Construct Grid. As for images usually cols && rows >> nCh we do not launch a whole thread-block in z dimension.
     * rowKernel must be of shape (W,1) (horizontal vector)
     * blockDim should be of shape: (X,Y), where X = 32, Y = TILE_HEIGHT_Y. 
     * @note: X= 16 or 32 to ensure coalesced memory access?? "To match coalescing requirements, it is
     * the best to set the width to 32 (or 16 on pre-Fermi cards)" from https://cg.ivd.kit.edu/downloads/GPGPU_assignment_3.pdf
     * 
     * gridDim should be of shape: ((A.mWidth / ROW_TILE_WIDTH + 1), A.mHeight, A.mChannels), resulting in one block per tile.
     * Hence, it is independet of threadsIdx (in contrast to all other kernel calls in cuCV).
     * A shared memory of size (W // 2 + ROW_TILE_WIDTH + W // 2) * sizeof(T1) + colKernel.mWidth * sizeof(T2) will be allocated dynamically.
     */
    {
        const size_t TILE_WIDTH_X = 32, TILE_HEIGHT_Y = 32, kNy = (size_t) colKernel.getHeight() / 2;   
        const dim3 threads(TILE_WIDTH_X, std::min(BLOCK_SIZE, 32));
        const dim3 blocks((A.getWidth() + TILE_WIDTH_X - 1) / TILE_WIDTH_X, (A.getHeight() + TILE_HEIGHT_Y - 1) / TILE_HEIGHT_Y, A.getNChannels());
        const size_t shCountsA = cuCV::convHelper::cucvAddBytes2Align((TILE_HEIGHT_Y + 2 * kNy) * TILE_WIDTH_X * sizeof(T1), sizeof(T2));
        const size_t shMemCounts = shCountsA + (size_t) (colKernel.getWidth() * colKernel.getHeight() * sizeof(T2));

        /// @todo: Check if block_size < colKernel.getHeight() / 2
        /// @todo: Warning or automatic reshape if kernel is not a col kernel
        if (colKernel.getHeight() <= 1 && colKernel.getWidth() > 1)
            fprintf(stderr, "Warning: Column Kernel height for seperated convolution is %d, but height is %d.\n", colKernel.getHeight(), colKernel.getWidth());

        ///< @todo is there another way to solve this? We need to synchronize threads on grid level in sepColConv2d if we want to pass OUT as input and output.
        ///< copy OUT as OUT serves as input and output. 
        ///< Kernel sepColConv2d() loads it into shared mem and writes it later. No synchronization is guaranteed.
        cuCV::CuMat<T1> temp(OUT); 
        
        cuCV::kernel::sepColConv2d<<<blocks, threads, shMemCounts>>>(OUT.kernel(), temp.kernel(), colKernel.kernel(), TILE_WIDTH_X, TILE_HEIGHT_Y, shCountsA / sizeof(T1), padding);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}


template <typename T1, typename T2>
cuCV::CuMat<T1> cuCV::sepSharedConv2d(const CuMat<T1> & A, const CuMat<T2> & rowKernel, const CuMat<T2> & colKernel, const cuCV::Padding padding) {
    
    // Determine size of output matrix
    int outW = 0, outH = 0;
    cuCV::convHelper::estimateOutSize(A, padding, &outW, &outH);

    // Create output matrix
    cuCV::CuMat<T1> out(outH, outW, A.getNChannels());
    out.allocateOnDevice();

    cuCV::sepSharedConv2d(out, A, rowKernel, colKernel, padding);

    return out;
}


template <typename T1, typename T2>
void cuCV::convHelper::checks(const CuMat<T1> & A, const CuMat<T2> & kernel) {
    if (A.getDataPtr() == NULL || kernel.getDataPtr() == NULL)
        throw cuCV::exception::NullPointer("PointerError: one or more operands point to NULL data!");

    if (A.getWidth() < kernel.getWidth() || A.getHeight() < kernel.getHeight()) {
        std::string msg = "DimensionError: kernel size is larger than matrix size: (" 
                + std::to_string(A.getWidth()) + ", " + std::to_string(A.getHeight()) + ") vs ("
                + std::to_string(kernel.getWidth()) + ", " + std::to_string(kernel.getHeight()) + ")";
        throw std::runtime_error(msg);
    }

    if (kernel.getHeight() % 2 == 0 || kernel.getWidth() % 2 == 0) {
        std::string msg = "Currently, only odd side lengths of filters are allowed in cuCV. You set: ("
            + std::to_string(kernel.getWidth()) + ", " + std::to_string(kernel.getHeight()) + ").";
        throw std::runtime_error(msg);
    }
}


template <typename T>
void cuCV::convHelper::estimateOutSize(const CuMat<T> & A, const Padding padding , int * width, int * height) {
    switch (padding) {
            case cuCV::Padding::ZERO:
                * width = A.getWidth(), * height = A.getHeight(); break;
            default:
                break;
        }
}


size_t cuCV::convHelper::cucvAddBytes2Align(size_t arrayCounts, size_t sizeofNextArrayType) {
    if (arrayCounts % sizeofNextArrayType == 0) 
        return arrayCounts;
    else {
        double tmp = (arrayCounts % sizeofNextArrayType);
        tmp = sizeofNextArrayType / tmp;
        CUCV_DEBUG_PRINT("Increasing size of array from: %ld to %ld, which is a multiple of %ld.\n", 
                arrayCounts, arrayCounts + (size_t) tmp, sizeofNextArrayType);
        return arrayCounts + (size_t) tmp;
    }
}




// template <typename T>
// cuCV::CuMat<T> cuCV::simpleConv2d(const CuMat<T> & A, const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY, const cuCV::Padding padding) {
//     // Create kernel
    
// }


/// Explicit template specialization
template void cuCV::add(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::add(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::add(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::add(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::add(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const CUCV_8U alpha);
template void cuCV::add(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const CUCV_16U alpha);
template void cuCV::add(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const CUCV_32F alpha);
template void cuCV::add(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const CUCV_64F alpha);

template void cuCV::dif(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::dif(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::dif(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::dif(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::dif(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const CUCV_8U alpha);
template void cuCV::dif(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const CUCV_16U alpha);
template void cuCV::dif(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const CUCV_32F alpha);
template void cuCV::dif(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const CUCV_64F alpha);

template void cuCV::mul(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::mul(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::mul(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::mul(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::mul(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const CUCV_8U alpha);
template void cuCV::mul(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const CUCV_16U alpha);
template void cuCV::mul(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const CUCV_32F alpha);
template void cuCV::mul(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const CUCV_64F alpha);

template void cuCV::div(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const cuCV::CuMat<CUCV_8U> & B);
template void cuCV::div(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const cuCV::CuMat<CUCV_16U> & B);
template void cuCV::div(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const cuCV::CuMat<CUCV_32F> & B);
template void cuCV::div(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const cuCV::CuMat<CUCV_64F> & B);

template void cuCV::div(cuCV::CuMat<CUCV_8U> & OUT, const cuCV::CuMat<CUCV_8U> & A, const CUCV_8U alpha);
template void cuCV::div(cuCV::CuMat<CUCV_16U> & OUT, const cuCV::CuMat<CUCV_16U> & A, const CUCV_16U alpha);
template void cuCV::div(cuCV::CuMat<CUCV_32F> & OUT, const cuCV::CuMat<CUCV_32F> & A, const CUCV_32F alpha);
template void cuCV::div(cuCV::CuMat<CUCV_64F> & OUT, const cuCV::CuMat<CUCV_64F> & A, const CUCV_64F alpha);

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

template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template void cuCV::sharedPaddingConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

template cuCV::CuMat<CUCV_8U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sharedPaddingConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sharedPaddingConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel, const cuCV::Padding padding);

template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_8U> & OUT, const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_16U> & OUT, const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_32F> & OUT, const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template void cuCV::sepSharedConv2d(cuCV::CuMat<CUCV_64F> & OUT, const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);

template cuCV::CuMat<CUCV_8U> cuCV::sepSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sepSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sepSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sepSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & rowKernel, const CuMat<CUCV_8U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sepSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sepSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sepSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sepSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & rowKernel, const CuMat<CUCV_16U> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sepSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sepSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sepSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sepSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & rowKernel, const CuMat<CUCV_32F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_8U> cuCV::sepSharedConv2d(const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_16U> cuCV::sepSharedConv2d(const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_32F> cuCV::sepSharedConv2d(const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);
template cuCV::CuMat<CUCV_64F> cuCV::sepSharedConv2d(const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & rowKernel, const CuMat<CUCV_64F> & colKernel, const cuCV::Padding padding);

// template cuCV::CuMat<CUCV_8U> cuCV::simpleConv2d(const CuMat<CUCV_8U> & A, const cuCV::Kernel kerneltype, const size_t kernelX, const size_t kernelY, const cuCV::Padding padding);

template void cuCV::convHelper::estimateOutSize(const CuMat<CUCV_8U> & A, const Padding padding , int * width, int * height);
template void cuCV::convHelper::estimateOutSize(const CuMat<CUCV_16U> & A, const Padding padding , int * width, int * height);
template void cuCV::convHelper::estimateOutSize(const CuMat<CUCV_32F> & A, const Padding padding , int * width, int * height);
template void cuCV::convHelper::estimateOutSize(const CuMat<CUCV_64F> & A, const Padding padding , int * width, int * height);

template void cuCV::convHelper::checks(const CuMat<CUCV_8U> & A, const CuMat<CUCV_8U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_16U> & A, const CuMat<CUCV_8U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_32F> & A, const CuMat<CUCV_8U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_64F> & A, const CuMat<CUCV_8U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_8U> & A, const CuMat<CUCV_16U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_16U> & A, const CuMat<CUCV_16U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_32F> & A, const CuMat<CUCV_16U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_64F> & A, const CuMat<CUCV_16U> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_8U> & A, const CuMat<CUCV_32F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_16U> & A, const CuMat<CUCV_32F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_32F> & A, const CuMat<CUCV_32F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_64F> & A, const CuMat<CUCV_32F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_8U> & A, const CuMat<CUCV_64F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_16U> & A, const CuMat<CUCV_64F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_32F> & A, const CuMat<CUCV_64F> & kernel);
template void cuCV::convHelper::checks(const CuMat<CUCV_64F> & A, const CuMat<CUCV_64F> & kernel);
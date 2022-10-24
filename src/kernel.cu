/**
 * @file kernel.cu
 * @author Vincent Hackstein (vinc37-git)
 * @brief 
 * @version 0.1
 * @date 2022-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "kernel.h"


template <typename T>
__global__ void cuCV::kernel::add(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    // threadIdx.x contains the index of the thread within the block
    // blockDim.x contains the size of thread block (number of threads in the thread block).

    // blockIdx.x contains the index of the block within the grid
    // gridDim.x contains the size of the grid

    //const int col = blockIdx.x * blockDim.x + threadIdx.x;
    //const int row = blockIdx.y * blockDim.y + threadIdx.y;
    //const int  ch = blockIdx.z * blockDim.z + threadIdx.z;

   const int index = blockIdx.y * blockDim.y + threadIdx.y * A.getWidth() 
        +  blockIdx.x * blockDim.x + threadIdx.x 
        + (A.getWidth()*A.getHeight()) * blockIdx.z * blockDim.z + threadIdx.z;  // linearisation of index
   // const int i_max = A.getSize();

    if (index < A.getSize())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] + B.getDataPtr()[index];
}


template <typename T>
__global__ void cuCV::kernel::add(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] + alpha;
}


template <typename T>
__global__ void cuCV::kernel::dif(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] - B.getDataPtr()[index];
}


template <typename T>
__global__ void cuCV::kernel::dif(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] - alpha;
}


template <typename T>
__global__ void cuCV::kernel::mul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] * B.getDataPtr()[index];
}


template <typename T>
__global__ void cuCV::kernel::mul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] * alpha;
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] / B.getDataPtr()[index];
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] / alpha;
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, float * pAlpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.getWidth() + col + (A.getWidth()*A.getHeight()) * ch;  // linearisation of index

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels())
        OUT.getDataPtr()[index] = A.getDataPtr()[index] / * pAlpha;
}


template <typename T> __global__ 
void cuCV::kernel::naiveMatmul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < OUT.getWidth() && row < OUT.getHeight() && ch < OUT.getNChannels()) {
        int OUTval = 0;

        for (int n = 0; n < A.getWidth(); n++)  // Matrix dimensions: MxN @ NxL
            OUTval += A.getDataPtr()[(row * A.getWidth() + n) + (A.getWidth()*A.getHeight())*ch] * B.getDataPtr()[(n * B.getWidth() + col) + (B.getWidth()*B.getHeight())*ch];
        
        OUT.getDataPtr()[(row * OUT.getWidth() + col) + (OUT.getWidth()*OUT.getHeight())*ch] = OUTval;
    }
}


template <typename T> __global__ 
void cuCV::kernel::matmul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int blockCh  = blockIdx.z;

    // Thread row and column within OUTsub
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    // Each thread block computes one sub-matrix OUTsub of OUT
    DeviceCuMat<T> OUTsub = OUT.getBlock(blockRow, blockCol, blockCh);

    // accumulate results in OUTval to set C_sub element
    T OUTval = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int n = 0; n < ((A.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE); n++) { // Matrix dimensions: MxN @ NxL
        // Get sub matrices of A and B
        DeviceCuMat<T> Asub = A.getBlock(blockRow, n);
        DeviceCuMat<T> Bsub = B.getBlock(n, blockCol);

        // Shared memory used to store elements of Asub und Bsub
        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Some entries in that matrix might be incorrect since the last submatrix might geht out of bounds of the full Matrix. Set those elems to 0.
        if (n * BLOCK_SIZE + threadCol < A.getWidth() && blockRow * BLOCK_SIZE + threadRow < A.getHeight()) {
            As[threadRow][threadCol] = Asub.getElement(threadRow, threadCol);
        } else {
            As[threadRow][threadCol] = 0.0;
        }
        if (n * BLOCK_SIZE + threadRow < B.getHeight() && blockCol * BLOCK_SIZE + threadCol < B.getWidth()) {
            Bs[threadRow][threadCol] = Bsub.getElement(threadRow, threadCol);
        } else {
            Bs[threadRow][threadCol] = 0.0;
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();    

        // Multiply Asub und Bsub together
        for (int elem = 0; (elem < BLOCK_SIZE); ++elem) {
            OUTval += As[threadRow][elem] * Bs[elem][threadCol];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    if (((blockCol * BLOCK_SIZE + threadCol) < OUT.getWidth()) && ((blockRow * BLOCK_SIZE + threadRow) < OUT.getHeight())) {
        OUTsub.setElement(threadRow, threadCol, OUTval);
    }
}


template <typename T> __device__
static T paddedValue(const int r, const int c, const cuCV::DeviceCuMat<T> A, const cuCV::Padding padding) {
    T returnVal = 0;
    switch (padding) {
        case cuCV::Padding::ZERO:
            return returnVal;                  
        default:
            return returnVal;
    }
}


template <typename T1, typename T2> __global__
void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding) {
/**
 * Notes about simpleConv2d:
 * - "Since the innermost processing loop of both row and column filter performs very few 
 * computations per iteration, the loop/branching overhead is very big, so in order to 
 * improve performance" the loops should be unrolled...
 * https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_64_website/projects/convolutionSeparable/doc/convolutionSeparable.pdf
 * 
 */
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int kNx = (kernel.getWidth() /*+ 1*/) / 2;
    const unsigned int kNy = (kernel.getHeight() /*+ 1*/) / 2;

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels()) {
        float out = 0;

        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = row - kNy, rK = kernel.getHeight() - 1; rK >= 0; ++r, --rK) {  // r is row of image. rK is row of kernel. rK decreases (kernel flip)
           for (int c = col - kNx, cK = kernel.getWidth() - 1; cK >= 0; ++c, --cK) {  // c is col of image. cK is col of kernel. rC decreases (kernel flip)        
                // Check if kernel overlaps with image edges
                if (r >= 0 && r < A.getHeight() && c >= 0 && c < A.getWidth()) {
                    out += (float) A.getElement(r, c, ch) * (float) kernel.getElement(rK, cK);  // accumulate
                }
                // index is out of bounds of A. Use Padding
                else {
                    out += (float) paddedValue(r, c, A, padding) * (float) kernel.getElement(rK, cK);
                }
            }
        }

        // Every Thread will insert an output value at its position in OUT.
        /// @todo Rounding would be nice for uint8 and uint16 outputs. However, we can not use typeid to determine type since it is device code. 
        OUT.setElement(row, col, ch, (T1) out);  
    }
}


template <typename T1, typename T2> __global__
void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding) {
    /**
     * Notes about simpleSharedConv2d():
     * - Loading kernel into __constant__ float could be an option, but: __constant__ memory has the 
     * lifetime of the CUDA context in which it is created.
     * - "However the problem with this approach is that elements on the boundaries will have neighbouring 
     * elements in other blocks and shared memory is only at the block level so it cannot be exploited for 
     * these elements. Therefore in these boundary cases we must read from global memory. 
     * However, the good news is that it is probable that these elements would have already been loaded to 
     * the L2 cache if they had been read into the shared memory of another block by the time this call is 
     * made. In this case it would actually have been read from the L2 cache rather than global memory which 
     * is faster." http://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html
     */

    /** simpleSharedConv2d() only allowed for kernel sizes < (BLOCK_SIZE, BLOCK_SIZE).
     * This must be checked before kernel call. */

    const unsigned int blockBoundLX = blockIdx.x * blockDim.x;   ///< Left bound of block: blockIdx.y * blockDim.y
    const unsigned int blockBoundUX = (blockIdx.x + 1) * blockDim.x - 1;  ///< Right bound: (blockIdx.y + 1) * blockDim.y (exclusive)
    const unsigned int blockBoundLY = blockIdx.y * blockDim.y;  ///< Upper bound: blockIdx.y * blockDim.y
    const unsigned int blockBoundUY = (blockIdx.y + 1) * blockDim.y - 1;  ///< Lower bound: (blockIdx.y + 1) * blockDim.y (exclusive)

    const unsigned int col = blockBoundLX + threadIdx.x;
    const unsigned int row = blockBoundLY + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int kNx = (kernel.getWidth() /*+ 1*/) / 2;
    const unsigned int kNy = (kernel.getHeight() /*+ 1*/) / 2;

    float out = 0;
    
    ///< Shared kernel per block. Row major to ensure coalesceding.
    ///< sharedKernel is dynamically allocated shared memory
    // reinterpret_cast() mechanism from https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ __align__(8) unsigned char sharedKernel_[];
    T2 * sharedKernel = reinterpret_cast<T2 *>(sharedKernel_);

    // fill sharedKernel with kernel. If blockDim.x * blockDim.y < kernel.getWidth() * kernel.getHeight(), loop over it.
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel.getWidth() * kernel.getHeight(); i += blockDim.x * blockDim.y) {
        sharedKernel[i] = kernel.getDataPtr()[i + kernel.getWidth() * kernel.getHeight() * threadIdx.z];  // apply channel offset to kernel.getDataPtr()
    }

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels()) {

        __shared__ T1 sharedAsub[BLOCK_SIZE][BLOCK_SIZE];
        sharedAsub[threadIdx.y][threadIdx.x] = A.getElement(row, col, ch);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();   

        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = row - kNy, rK = kernel.getHeight() - 1; rK >= 0; ++r, --rK) {  // r is row of image. rK is row of kernel. rK decreases (kernel flip)
           for (int c = col - kNx, cK = kernel.getWidth() - 1; cK >= 0; ++c, --cK) {  // c is col of image. cK is col of kernel. rC decreases (kernel flip)        
                // Check if kernel overlaps with image edges
                if (r >= 0 && r < A.getHeight() && c >= 0 && c < A.getWidth()) {
                    if (r < blockBoundLY || r > blockBoundUY || c < blockBoundLX || c > blockBoundUX) {  // kernel is partially outside of block
                        // To convolute with values outside of current block, we need to load those values from global memory.
                        // They might be loaded in L2 chache already though since they were loaded into shared memory of another block before.
                        out += (float) A.getElement(r, c, ch) * (float) sharedKernel[rK * kernel.getWidth() + cK];  // accumulate from global
                    }
                    else {
                        // kernel is inside of block
                        out += (float) sharedAsub[r - blockBoundLY][c - blockBoundLX] * (float) sharedKernel[rK * kernel.getWidth() + cK];  // accumulate from shared 
                    }
                }
                // index is out of bounds of A. Use Padding
                else {
                    out += (float) paddedValue(r, c, A, padding) * (float) sharedKernel[rK * kernel.getWidth() + cK];
                }
            }
        }
        // Every Thread will insert an output value at its position in OUT.
        /// @todo Rounding would be nice for uint8 and uint16 outputs. However, we can not use typeid to determine type since it is device code. 
        OUT.setElement(row, col, ch, (T1) out);  
    }
}


template <typename T1, typename T2> __global__
void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const size_t shElemsA, const cuCV::Padding padding) {
    /**
     * Notes about sharedPaddingConv2d():
     * - Load image block/tile plus apron into shared memory. 
     *  After the image is stored in shared memory, load the filter.
     */

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int kNx = (kernel.getWidth() /*+ 1*/) / 2;
    const unsigned int kNy = (kernel.getHeight() /*+ 1*/) / 2;

    float out = 0;
    
    /** Shared kernel per block. Row major to ensure coalesceding.
    * sharedMem_ is dynamically allocated shared memory
    * and will be split into two arrays of different types T1 and T2
    * size must match (blockDim.x * 3) * (blockDim.y * 3) * sizeof(T1) + kernel.width * kernel.height * sizeof(T2)
    * reinterpret_cast() mechanism from https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    */
    extern __shared__ __align__(8) unsigned char sharedMem_[];
    T1 * sharedA = reinterpret_cast<T1 *>(sharedMem_);
    T2 * sharedK = reinterpret_cast<T2 *>(& sharedA[shElemsA]);    

    // Divide image into 9 squares around the current tile. use every thread to load 9 values sequentially
    // unroll loop since it's always a loop of 3x3 iterations.
#pragma unroll
    for (int rS = 0; rS < 3; ++rS) {  // rS is the square in Y, r and c are row/column of image.
        for (int cS = 0; cS < 3; ++cS) {
            int c = (cS-1) * blockDim.x + col;
            int r = (rS-1) * blockDim.y + row;
            int shStrideX = 3 * blockDim.x;
            int iSharedA = threadIdx.y * shStrideX  // increment per row
                    + threadIdx.x  // increment per col
                    + cS * blockDim.x  // offset per square in col-dir
                    + rS * blockDim.y * blockDim.y * 3;  // offset per square in row-dir 

            if (c < 0 || r < 0 || r >= A.getHeight() || c >= A.getWidth()) { // out of bounds. use padding
                sharedA[iSharedA] = paddedValue(r, c, A, padding);
            }
            else {  // inbound
                cuCV::DeviceCuMat<T1> Asub = A.getBlock(blockIdx.y + rS-1, blockIdx.x + cS-1, ch);
                sharedA[iSharedA] = Asub.getElement(threadIdx.y, threadIdx.x, ch);
            } 
        }
    }    
    // fill sharedK with kernel. If blockDim.x * blockDim.y < kernel.getWidth() * kernel.getHeight(), loop over it.
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < kernel.getWidth() * kernel.getHeight(); i += blockDim.x * blockDim.y) {
        sharedK[i] = kernel.getDataPtr()[i + kernel.getWidth() * kernel.getHeight() * threadIdx.z];  // apply channel offset to kernel.getDataPtr()
    }

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads(); 

    if (col < A.getWidth() && row < A.getHeight() && ch < A.getNChannels()) {
        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = blockDim.y - kNy + threadIdx.y, rK = kernel.getHeight() - 1; rK >= 0; ++r, --rK) {  // r is row of 3*3 tiles. rK is row of kernel. rK decreases (kernel flip)
           for (int c = blockDim.x - kNx + threadIdx.x, cK = kernel.getWidth() - 1; cK >= 0; ++c, --cK) {  // c is col of 3*3 tiles. cK is col of kernel. rC decreases (kernel flip)         
                out += (float) sharedA[r * blockDim.x * 3 + c] * (float) sharedK[rK * kernel.getWidth() + cK];
            }
        }
        // Every Thread will insert an output value at its position in OUT.
        /// @todo Rounding would be nice for uint8 and uint16 outputs. However, we can not use typeid to determine type since it is device code. 
        OUT.setElement(row, col, ch, (T1) out);          
    }
}



template <typename T1, typename T2> __global__
void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding) {
    /**
     * rowKernel must be of shape (W,1) (horizontal/row vector)
     * blockDim should be of shape: (X,1,Z), where X = KERNEL_HALF_ALIGNED + ROW_TILE_W + KERNEL_HALF and Z the depth of the image.
     * gridDim should be of shape: ((A.getWidth() / ROW_TILE_WIDTH + 1), A.getHeight(), A.getNChannels()), resulting in one block per tile.
     * Hence, it is independet of threadsIdx (in contrast to all other kernel calls in cuCV).
     * A shared memory of size (W // 2 + ROW_TILE_WIDTH + W // 2) * sizeof(T1) + rowKernel.getWidth() * sizeof(T2) will be allocated dynamically.
     * Size Instruction must come from host.
     * Data of the tile and apron (overlap of tiles during convolution) on both sides is loaded into shared memory.
     * @todo: Align data to meet half warp requirement ? If yes, the first n threads will be inactive to make subsequent threads aligned properly
     * Sync threads 
     * Compute convolution for ROW_TILE and save in OUT
     * back on host: call sepConv2d
     */

    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int kNx = (unsigned int) rowKernel.getWidth() / 2;  ///< Truncated half of kernel width (or "kernel radius")
    const unsigned int ALIGNED_OFFSET = blockDim.x - tileWidth - 2 * kNx;  ///< This may be less or equal to kNx in order to align the data.

    const unsigned int col0 = blockIdx.x * tileWidth;  ///< or x0: The index where the actual data of the tile begins.
    const unsigned int colN = col0 + tileWidth - 1;  ///< or xN: The index where the data of the tile ends.
    const int col0Apron = col0 - kNx;  ///< The index where the apron begins. Note that it may be negative (out of bounds to the left).
    const int col0ApronAligned = col0Apron - ALIGNED_OFFSET;  ///< The index of A that will meet half warp requirement.

    const unsigned int row0 = blockIdx.y;  ///> The index of the row of currents block.

    //cuCV::DeviceCuMat rowMat = A.getSubCuMat(row0, col0, ch, blockDim.x, 1);
    
    /// Cast dynamic shared memory to T1 and T2 respectively
    extern __shared__ __align__(8) unsigned char sharedMem_[];
    T1 * sharedA = reinterpret_cast<T1 *>(sharedMem_);
    T2 * sharedK = reinterpret_cast<T2 *>(& sharedA[shElemsA]); 
    //T2 * sharedK = reinterpret_cast<T2 *>(& sharedA[tileWidth + kNx * 2]); 

    /** Load data of A (__global__) into sharedA.
     * Skip the ALIGNED_OFFSET offset, but load the apron. If the apron is out of bounds, get padded value.
     * If A.getWidth() and tileWidth meet half-warp requirements (hence, are multiple of half warp size),
     * row0 + col0ApronAligned should also be a multuiple of half-warp size. This will ensure proper 
     * alignment for coalesced data read.
     */
    int col = col0ApronAligned + threadIdx.x;  // col will be aligned for threadId.x == 0
    if (col >= col0Apron) {  // some thread which point to data out of bounds of tile and apron will be inactive
        const int iSharedA = col - col0Apron;  
        if (col >= 0 && col < A.getWidth())  // inbound
            sharedA[iSharedA] = A.getElement(row0, col, threadIdx.z);
        else  // outbound
            sharedA[iSharedA] = paddedValue(row0, col, A, padding);
    }

    /** Load data of rowKernel (__global__) into sharedK.
     * If there ARE inactive threads due to alignemt, 
     * (some) of them will be used to load the kernel
     * @bug use for loop for case length of sharedK > blockDim
     */
    if (threadIdx.x < rowKernel.getWidth())
        sharedK[threadIdx.x] = rowKernel.getElement(0, threadIdx.x, threadIdx.z);

    // Synchronize threads
    __syncthreads();

    /** Convolute kernel over row. 
     * The result will be loaded back to __global__ memory of OUT. 
     * col is the column of A and OUT. 
     * It must be shifted by col0Apron when accessing data from shared 
     * memory to make it relative to shared memory start.
     */
    float out = 0;
    col = col0 + threadIdx.x;
    if (col < (int) colN && col < A.getWidth()) {
        for (int i = col - col0Apron - kNx, iK = rowKernel.getWidth() - 1; iK >= 0; --iK, ++i) {
            out += sharedA[i] * sharedK[iK];
        }

        OUT.setElement(row0, col, ch, (T1) out);
    }
}


template <typename T1, typename T2> __global__
void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> colKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding) {
    /**
     * colKernel must be of shape (1,Y): vertical / column vector
     * blockDim should be of shape: (X,Y,Z), where 
     *      X = tileWidth : Should be of length 32 at least: in case of uint8 pixels it will result in 32 threads (warp)
     *      Y = KERNEL_HALF + tileHeight + KERNEL_HALF : 
     *      Z = the depth of the image.
     * gridDim should be of shape: (gX, gY, gZ), where
     *      gX = A.getWidth() / tileWidth + 1
     *      gY = A.getHeight() / tileHeight + 1
     *      gZ = A.mChannel
     * A shared memory of size: (blockDim.y * blockDim.x) * sizeof(T1) + colKernel.getWidth() * sizeof(T2) will be allocated dynamically.
     * Size Instruction must come from host.
     * Data of the tile and apron (overlap of tiles during convolution) on both sides is loaded into shared memory.
     * Sync threads 
     * Compute convolution for col tile and ACCUMULATE IT with / on (?) OUT
     */

    const unsigned int kNy = (unsigned int) colKernel.getHeight() / 2;  ///< Truncated half of kernel height (or "kernel radius")

    // row indices
    const unsigned int row0 = blockIdx.y * tileHeight;  ///< or y0: The index where the actual data of the tile begins.
    const unsigned int rowN = row0 + tileHeight - 1;  ///< or yN: The index where the data of the tile ends.
    const int row0Apron = row0 - kNy;  ///< The index where the apron begins. Note that it may be negative (out of bounds to the top).
    const unsigned int rowNApron = rowN + kNy;  ///< The index where the apron ends. Note that it may be out of bounds to the bottom).

    // col indices
    const unsigned int col0 = blockIdx.x * tileWidth;  ///< The index of the row of currents block.
    //const unsigned int colN = col0 + tileWidth - 1;  ///< The index of the last row of currents block.

    // channel index
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;  ///< index of current channel.
    
    //cuCV::DeviceCuMat Asub = A.getSubCuMat(row0Apron, col0, ch, tileWidth, tileHeight);  // That is the current tile inclusive apron

    /// Cast dynamic shared memory to T1 and T2 respectively
    extern __shared__ __align__(8) unsigned char sharedMem_[];
    T1 * sharedA = reinterpret_cast<T1 *>(sharedMem_);
    T2 * sharedK = reinterpret_cast<T2 *>(& sharedA[shElemsA]); 

    // Thread indices on image
    const unsigned int col = col0 + threadIdx.x;

    /** Load data of A (__global__) into sharedA in row major order.
     * Load the apron, too. If the apron is out of bounds, get padded value.
     * if blockDim.x * blockDim.y < tileWidth * tileHeightApron, the data is loaded in blocks
     * where each thread loads one pixel per iteration.
     */
    unsigned int iSharedA = threadIdx.y * tileWidth + threadIdx.x;

    for (int row = row0Apron + threadIdx.y; row <= (int) rowNApron /*&& row < A.getHeight()*/; ) {

        if (col < A.getWidth() && row >= 0 && row < A.getHeight()) {  // inbound. @note: col is unsigned.
            sharedA[iSharedA] = A.getElement(row, col, ch);
        }
        else {  //outbound
            sharedA[iSharedA] = (T1) paddedValue(row, col, A, padding);
        }
        row += blockDim.y;  // increse row by blockDim.y
        iSharedA += blockDim.y * blockDim.x;
    }

    /** Load data of colKernel (__global__) into sharedK. 
     * @bug use for loop for case length of sharedK > blockDim */
    int iSharedK = threadIdx.y * blockDim.x + threadIdx.x;
    if (iSharedK < colKernel.getHeight())
        sharedK[iSharedK] = colKernel.getElement(0, iSharedK, threadIdx.z);

    __syncthreads();

    /** Convolute kernel over columns. 
     * The result will be loaded back to __global__ memory of OUT. 
     * col is the column of A and OUT. 
     * It must be shifted by col0Apron when accessing data from shared 
     * memory to make it relative to shared memory start.
     */
    
    for (int row = row0 + threadIdx.y; row <= (int) rowN && row < A.getHeight(); ) {
        float out = 0;

        if (col < A.getWidth() && row < A.getHeight()) {
            unsigned int iSharedA = (threadIdx.y + kNy) * tileWidth + threadIdx.x;  // iSharedA should not start at 0 due to the apron

            for (int i = iSharedA - kNy * tileWidth, iK = colKernel.getHeight() - 1; iK >= 0; --iK, i+=tileWidth) {
                out += sharedA[i] * sharedK[iK];
            }
            
            OUT.setElement(row, col, ch, (T1) out);
        }

        row += blockDim.y;  // increse row by blockDim.y
        iSharedA += blockDim.y * blockDim.x;
    }
}


template <typename T> __global__
void cuCV::kernel::zeros(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.getWidth() + col + (OUT.getWidth()*OUT.getHeight()) * ch;  // linearisation of index

    if (col < OUT.getWidth() && row < OUT.getHeight() && ch < OUT.getNChannels())
        OUT.getDataPtr()[index] = 0;
}


template <typename T> __global__
void cuCV::kernel::ones(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.getWidth() + col + (OUT.getWidth()*OUT.getHeight()) * ch;  // linearisation of index

    if (col < OUT.getWidth() && row < OUT.getHeight() && ch < OUT.getNChannels())
        OUT.getDataPtr()[index] = 1;
}


template <typename T> __global__
void cuCV::kernel::eye(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.getWidth() + col + (OUT.getWidth()*OUT.getHeight()) * ch;  // linearisation of index

    if (col < OUT.getWidth() && row < OUT.getHeight() && ch < OUT.getNChannels()) {
        if (col == row)
            OUT.getDataPtr()[index] = 1;
        else
            OUT.getDataPtr()[index] = 0;
    }
}


__device__
static double gaussian1dDevice(double x, double mu, double sigma) {
    const double a = (x - mu) / sigma;
    return std::exp(-0.5 * a * a);    
}


template <typename T> __global__ 
void cuCV::kernel::gauss(cuCV::DeviceCuMat<T> OUT, double sigma, bool norm, float * sum) {
    ///< @note: Only squared matrices are allowed and length of sides must the an odd number of elements. 

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < OUT.getWidth() && row < OUT.getHeight() && ch < OUT.getNChannels()) {
        size_t radiusX = (size_t) OUT.getWidth() / 2;
        size_t radiusY = (size_t) OUT.getHeight() / 2;

        // Calculate gaussian distributed value for current position / thread
        double out = gaussian1dDevice(row, radiusY, sigma) * gaussian1dDevice(col, radiusX, sigma);

        // add out value to sum.
        /// @bug Check if we need a lock since multiple threads could write at same time.
        /// @note atomic function: performed in one atomic transaction
        if (ch == 0)
            atomicAdd(sum, out);

        OUT.setElement(row, col, ch, (T) out);
    }
}



/// Explicit template specialization
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const CUCV_32F alpha);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const CUCV_32F alpha);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const CUCV_32F alpha);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const CUCV_32F alpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, float * pAlpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, float * pAlpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, float * pAlpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, float * pAlpha);

template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_32F> OUT, const DeviceCuMat<CUCV_32F> A, const DeviceCuMat<CUCV_32F> B);
template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);

template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleSharedConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);

template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_32F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sharedPaddingConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const size_t shElemsA, const cuCV::Padding padding);

template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepRowConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t shElemsA, const cuCV::Padding padding);

template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_32F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_32F> OUT, const cuCV::DeviceCuMat<CUCV_32F> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);
template __global__ void cuCV::kernel::sepColConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> rowKernel, const size_t tileWidth, const size_t tileHeight, const size_t shElemsA, const cuCV::Padding padding);

template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_32F> OUT);
template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_64F> OUT);

template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_32F> OUT);
template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_64F> OUT);

template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_32F> OUT);
template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_64F> OUT);

//template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_8U> OUT, double sigma, bool norm, float * sum);
//template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_16U> OUT, double sigma, bool norm, float * sum);
template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_32F> OUT, double sigma, bool norm, float * sum);
template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_64F> OUT, double sigma, bool norm, float * sum);
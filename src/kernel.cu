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

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] + B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::add(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] + alpha;
}


template <typename T>
__global__ void cuCV::kernel::dif(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] - B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::dif(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] - alpha;
}


template <typename T>
__global__ void cuCV::kernel::mul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] * B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::mul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] * alpha;
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] / B.mData[index];
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const T alpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] / alpha;
}


template <typename T>
__global__ void cuCV::kernel::div(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, float * pAlpha) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels)
        OUT.mData[index] = A.mData[index] / * pAlpha;
}


template <typename T> __global__ 
void cuCV::kernel::naiveMatmul(DeviceCuMat<T> OUT, const DeviceCuMat<T> A, const DeviceCuMat<T> B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels) {
        int OUTval = 0;

        for (int n = 0; n < A.mWidth; n++)  // Matrix dimensions: MxN @ NxL
            OUTval += A.mData[(row * A.mWidth + n) + (A.mWidth*A.mHeight)*ch] * B.mData[(n * B.mWidth + col) + (B.mWidth*B.mHeight)*ch];
        
        OUT.mData[(row * OUT.mWidth + col) + (OUT.mWidth*OUT.mHeight)*ch] = OUTval;
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
    DeviceCuMat<T> OUTsub = OUT.getSubCuMat(blockRow, blockCol, blockCh);

    // accumulate results in OUTval to set C_sub element
    T OUTval = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int n = 0; n < ((A.mWidth + BLOCK_SIZE - 1) / BLOCK_SIZE); n++) { // Matrix dimensions: MxN @ NxL
        // Get sub matrices of A and B
        DeviceCuMat<T> Asub = A.getSubCuMat(blockRow, n);
        DeviceCuMat<T> Bsub = B.getSubCuMat(n, blockCol);

        // Shared memory used to store elements of Asub und Bsub
        __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Some entries in that matrix might be incorrect since the last submatrix might geht out of bounds of the full Matrix. Set those elems to 0.
        if (n * BLOCK_SIZE + threadCol < A.mWidth && blockRow * BLOCK_SIZE + threadRow < A.mHeight) {
            As[threadRow][threadCol] = Asub.getElement(threadRow, threadCol);
        } else {
            As[threadRow][threadCol] = 0.0;
        }
        if (n * BLOCK_SIZE + threadRow < B.mHeight && blockCol * BLOCK_SIZE + threadCol < B.mWidth) {
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
    if (((blockCol * BLOCK_SIZE + threadCol) < OUT.mWidth) && ((blockRow * BLOCK_SIZE + threadRow) < OUT.mHeight)) {
        OUTsub.setElement(threadRow, threadCol, OUTval);
        //printf("Set (%d, %d, %d), stride: %d = %d ==/!= %d\n", threadRow, threadCol, threadIdx.z, OUT.mStrideX, OUTval, OUTsub.getElement(threadRow, threadCol));
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

    const unsigned int Nx = (kernel.mWidth /*+ 1*/) / 2;
    const unsigned int Ny = (kernel.mHeight /*+ 1*/) / 2;

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels) {
        double out = 0;

        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = row - Ny, rK = kernel.mHeight - 1; rK >= 0; ++r, --rK) {  // r is row of image. rK is row of kernel. rK decreases (kernel flip)
           for (int c = col - Nx, cK = kernel.mWidth - 1; cK >= 0; ++c, --cK) {  // c is col of image. cK is col of kernel. rC decreases (kernel flip)        
                // Check if kernel overlaps with image edges
                if (r >= 0 && r < A.mHeight && c >= 0 && c < A.mWidth) {
                    out += A.getElement(r, c, ch) * kernel.getElement(rK, cK);  // accumulate
                }
                // index is out of bounds of A. Use Padding
                else {
                    out += paddedValue(r, c, A, padding) * kernel.getElement(rK, cK);
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

    const unsigned int Nx = (kernel.mWidth /*+ 1*/) / 2;
    const unsigned int Ny = (kernel.mHeight /*+ 1*/) / 2;

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels) {
        double out = 0;
        
        ///< Shared kernel per block. Row major to ensure coalesceding.
        ///< sharedKernel is dynamically allocated shared memory
        // reinterpret_cast() mechanism from https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
        extern __shared__ __align__(sizeof(T2)) unsigned char sharedKernel_[];
        T2 * sharedKernel = reinterpret_cast<T2 *>(sharedKernel_);

        if (threadIdx.x < kernel.mWidth && threadIdx.y < kernel.mHeight)
            sharedKernel[threadIdx.y * kernel.mWidth + threadIdx.x] = kernel.getElement(threadIdx.y, threadIdx.x, threadIdx.z);

        __shared__ T1 sharedAsub[BLOCK_SIZE][BLOCK_SIZE];
        sharedAsub[threadIdx.y][threadIdx.x] = A.getElement(row, col, ch);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();   

        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = row - Ny, rK = kernel.mHeight - 1; rK >= 0; ++r, --rK) {  // r is row of image. rK is row of kernel. rK decreases (kernel flip)
           for (int c = col - Nx, cK = kernel.mWidth - 1; cK >= 0; ++c, --cK) {  // c is col of image. cK is col of kernel. rC decreases (kernel flip)        
                // Check if kernel overlaps with image edges
                if (r >= 0 && r < A.mHeight && c >= 0 && c < A.mWidth) {
                    if (r < blockBoundLY || r > blockBoundUY || c < blockBoundLX || c > blockBoundUX) {  // kernel is partially outside of block
                        // To convolute with values outside of current block, we need to load those values from global memory.
                        // They might be loaded in L2 chache already though since they were loaded into shared memory of another block before.
                        out += A.getElement(r, c, ch) * sharedKernel[rK * kernel.mWidth + cK];  // accumulate from global
                    }
                    else {
                        // kernel is inside of block
                        out += sharedAsub[r - blockBoundLY][c - blockBoundLX] * sharedKernel[rK * kernel.mWidth + cK];  // accumulate from shared 
                        ///< @bug [threadIdx.y][threadIdx.x] always same
                    }
                }
                // index is out of bounds of A. Use Padding
                else {
                    out += paddedValue(r, c, A, padding) * sharedKernel[rK * kernel.mWidth + cK];
                }
            }
        }
        // Every Thread will insert an output value at its position in OUT.
        /// @todo Rounding would be nice for uint8 and uint16 outputs. However, we can not use typeid to determine type since it is device code. 
        OUT.setElement(row, col, ch, (T1) out);  
    }

}

template <typename T> __global__
void cuCV::kernel::zeros(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.mWidth + col + (OUT.mWidth*OUT.mHeight) * ch;  // linearisation of index

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels)
        OUT.mData[index] = 0;
}


template <typename T> __global__
void cuCV::kernel::ones(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.mWidth + col + (OUT.mWidth*OUT.mHeight) * ch;  // linearisation of index

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels)
        OUT.mData[index] = 1;
}


template <typename T> __global__
void cuCV::kernel::eye(cuCV::DeviceCuMat<T> OUT) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * OUT.mWidth + col + (OUT.mWidth*OUT.mHeight) * ch;  // linearisation of index

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels) {
        if (col == row)
            OUT.mData[index] = 1;
        else
            OUT.mData[index] = 0;
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

    if (col < OUT.mWidth && row < OUT.mHeight && ch < OUT.mChannels) {
        size_t radius = (size_t) OUT.mWidth / 2;

        // Calculate gaussian distributed value for current position / thread
        double out = gaussian1dDevice(row, radius, sigma) * gaussian1dDevice(col, radius, sigma);

        // add out value to sum.
        /// @bug Check if we need a lock since multiple threads could write at same time.
        /// @note atomic function: performed in one atomic transaction
        if (ch == 0)
            float old = atomicAdd(sum, out);

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
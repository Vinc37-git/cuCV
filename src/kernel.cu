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


template <typename T1, typename T2> __global__
void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<T1> OUT, const cuCV::DeviceCuMat<T1> A, const cuCV::DeviceCuMat<T2> kernel, const cuCV::Padding padding) {
    // const unsigned int blockRow = blockIdx.y;
    // const unsigned int blockCol = blockIdx.x;
    // const unsigned int blockCh  = blockIdx.z;

    // const unsigned int threadRow = threadIdx.y;
    // const unsigned int threadCol = threadIdx.x;

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int  ch = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int index = row * A.mWidth + col + (A.mWidth*A.mHeight) * ch;  // linearisation of index

    const unsigned int kernelHalfWidth = (kernel.mWidth /*+ 1*/) / 2;
    const unsigned int kernelHalfHeight = (kernel.mHeight /*+ 1*/) / 2;

    if (col < A.mWidth && row < A.mHeight && ch < A.mChannels) {
        double out = 0;

        /// @todo First we will assume the kernel hasn a odd number of cols and rows
        for (int r = row-kernelHalfHeight, rK = 0; rK < kernel.mHeight; ++r, ++rK) {  // r is row of image. rK is row of kernel
            for (int c = col-kernelHalfWidth, cK = 0; cK < kernel.mWidth; ++c, ++cK) {  // c is col of image. cK is col of kernel

                // Check if kernel overlaps with image edges
                if (r >= 0 && r < A.mHeight && c >= 0 && c < A.mWidth) {
                    out += A.getElement(r, c, ch) * kernel.getElement(rK, cK);  // accumulate
                }
                // index is out of bounds of A. Use Padding
                else {
                    switch (padding) {
                        case cuCV::Padding::ZERO:
                            // out += 0;
                            continue;                    
                        default:
                            continue;
                    }
                }
            }
        }

        // Every Thread will insert an output value at its position in OUT.
        /// @todo Rounding would be nice for uint8 and uint16 outputs. However, we can not use typeid to determine type since it is device code. 
        OUT.mData[index] = (T1) out;  
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
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::add(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::dif(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::mul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const CUCV_8U alpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const CUCV_16U alpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const CUCV_64F alpha);

template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, float * pAlpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, float * pAlpha);
template __global__ void cuCV::kernel::div(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, float * pAlpha);

template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::naiveMatmul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_8U> OUT, const DeviceCuMat<CUCV_8U> A, const DeviceCuMat<CUCV_8U> B);
template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_16U> OUT, const DeviceCuMat<CUCV_16U> A, const DeviceCuMat<CUCV_16U> B);
template __global__ void cuCV::kernel::matmul(DeviceCuMat<CUCV_64F> OUT, const DeviceCuMat<CUCV_64F> A, const DeviceCuMat<CUCV_64F> B);

template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_8U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_16U> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_8U> OUT, const cuCV::DeviceCuMat<CUCV_8U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_16U> OUT, const cuCV::DeviceCuMat<CUCV_16U> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);
template __global__ void cuCV::kernel::simpleConv2d(cuCV::DeviceCuMat<CUCV_64F> OUT, const cuCV::DeviceCuMat<CUCV_64F> A, const cuCV::DeviceCuMat<CUCV_64F> kernel, const cuCV::Padding padding);

template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::zeros(cuCV::DeviceCuMat<CUCV_64F> OUT);

template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::ones(cuCV::DeviceCuMat<CUCV_64F> OUT);

template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_8U> OUT);
template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_16U> OUT);
template __global__ void cuCV::kernel::eye(cuCV::DeviceCuMat<CUCV_64F> OUT);

template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_8U> OUT, double sigma, bool norm, float * sum);
template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_16U> OUT, double sigma, bool norm, float * sum);
template __global__ void cuCV::kernel::gauss(cuCV::DeviceCuMat<CUCV_64F> OUT, double sigma, bool norm, float * sum);
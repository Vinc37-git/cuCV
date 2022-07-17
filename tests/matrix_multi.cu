#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1000
#define ROWS 15000//16384
#define COLS 15000
#define MAX_ERR 1e-6
#define BLOCK_SIZE 32

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

size_t ij(int i, int j){
    // Row Major
    return i * ROWS + j;
}

void print_mat(Matrix M) {
    for (int i = 0; i < M.height; i++) {
        if (i < 4) {
            for (int j = 0; j < M.width; j++) {
                if (j < 4) {
                    printf("%4.0f ", M.elements[ij(i,j)]);
                }
            }
        printf("\n");
        }
    }
}

// __global__ void vector_add(float *out, float *a, float *b, int n) {
//     // threadIdx.x contains the index of the thread within the block
//     // blockDim.x contains the size of thread block (number of threads in the thread block).

//     // blockIdx.x contains the index of the block with in the grid
//     // gridDim.x contains the size of the grid

//     int index = (blockIdx.x * blockDim.x) + threadIdx.x;  // linearisation of index tuple
//     int stride = gridDim.x * blockDim.x;  // 
//     for(int i = index /*"range" for every open thread*/; i < n; i += stride /* e.g + 256*/){
//         out[i] = a[i] + b[i];
//     }
// }

// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//     #define printf(f, ...) ((void)(f, __VA_ARGS__),0) 
// #endif 

__global__ void matrix_multi_elemwise(Matrix OUT, const Matrix A, const Matrix B) {
    // NOTE: Generally memory allocated dynamically on device (GPU) and 
    // we cannot use two-dimensional indices (e.g. A[row][column]) 
    // to access matrices -> linear indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * A.width + col;  // linearisation of index

    if (col < A.width && row < A.height) {
        OUT.elements[index] = A.elements[index] * B.elements[index];
    }
}

// WITHOUT SHARED MEMORY. TAKES 96.8s for ROWS = COLS = 15000
__global__ void matrix_multi_not_shared(Matrix OUT, const Matrix A, const Matrix B) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < OUT.width && row < OUT.height) {
        int OUT_value = 0;

        for (int n = 0; n < A.width; n++) { // Matrix dimensions: MxN @ NxL
            OUT_value += A.elements[row * A.width + n] * B.elements[n * B.width + col];
        }
        OUT.elements[row * OUT.width + col] = OUT_value;
    }
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix get_submatrix(Matrix M, int block_id_row, int block_id_col) {
    Matrix M_sub;
    M_sub.width = BLOCK_SIZE;
    M_sub.height = BLOCK_SIZE;
    M_sub.stride = M.stride;
    M_sub.elements = &M.elements[M.stride * block_id_row * BLOCK_SIZE + block_id_col * BLOCK_SIZE];
    return M_sub;
}

// Set a specific Matrix element
__device__ void set_element(Matrix M, const int row, const int col, float val) {
    M.elements[M.stride * row + col] = val;
}

// Get a specific Matrix element
__device__ float get_element(const Matrix M, int row, int col) {
    return M.elements[M.stride * row + col];
}

// // Forward declaration of the matrix multiplication kernel
// __global__ void matrix_multi(Matrix, const Matrix, const Matrix);

// WITH SHARED MEMORY: TAKES: 19.4379s WITH BLOCK SIZE of 32 x 32
__global__ void matrix_multi(Matrix OUT, const Matrix A, const Matrix B) {
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Thread row and column within OUT_sub
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Each thread block computes one sub-matrix OUT_sub of OUT
    Matrix OUT_sub = get_submatrix(OUT, block_row, block_col);

    // accumulate results in OUT_value to set C_sub element
    float OUT_value = 0;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int n = 0; n < ((A.width + BLOCK_SIZE - 1) / BLOCK_SIZE); n++) { // Matrix dimensions: MxN @ NxL
        // Get sub matrices of A and B
        Matrix A_sub = get_submatrix(A, block_row, n);
        Matrix B_sub = get_submatrix(B, n, block_col);

        // Shared memory used to store elements of A_sub und B_sub
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Some entries in that matrix might be incorrect since the last submatrix might geht out of bounds of the full Matrix. Set those elems to 0.
        if (n * BLOCK_SIZE + thread_col < A.width && block_row * BLOCK_SIZE + thread_row < A.height) {
            As[thread_row][thread_col] = get_element(A_sub, thread_row, thread_col);
        } else {
            As[thread_row][thread_col] = 0.0;
        }
        if (n * BLOCK_SIZE + thread_row < B.height && block_col * BLOCK_SIZE + thread_col < B.width) {
            Bs[thread_row][thread_col] = get_element(B_sub, thread_row, thread_col);
        } else {
            Bs[thread_row][thread_col] = 0.0;
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();    

        // Multiply A_sub und B_sub together
        for (int elem = 0; (elem < BLOCK_SIZE); ++elem) {
            OUT_value += As[thread_row][elem] * Bs[elem][thread_col];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    if (((block_col * BLOCK_SIZE + thread_col) < OUT.width) && ((block_row * BLOCK_SIZE + thread_row) < OUT.height)) {
        set_element(OUT_sub, thread_row, thread_col, OUT_value);
    }
}

int main(){
    printf("Start\n");
    Matrix A, B, OUT;
    Matrix dev_A, dev_B, dev_OUT; 

    size_t SIZE = ROWS * COLS * sizeof(float);

    // Allocate host memory
    A.elements = (float*) malloc(SIZE);
    B.elements = (float*) malloc(SIZE);
    OUT.elements = (float*) malloc(SIZE);

    // Initialize host matrices
    A.height = ROWS; A.width = A.stride = COLS;
    B.height = ROWS; B.width = B.stride = COLS;
    OUT.height = A.height; OUT.width = OUT.stride = B.width;

    for (int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++){
            A.elements[ij(i, j)] = 2.0f;
            B.elements[ij(i, j)] = 3.0f;
        }
    }

    // Allocate device memory
    cudaMalloc((void**) &dev_A.elements, SIZE);
    cudaMalloc((void**) &dev_B.elements, SIZE);
    cudaMalloc((void**) &dev_OUT.elements, SIZE);

    dev_A.height = A.height; dev_A.width = A.width, dev_A.stride = A.stride; 
    dev_B.height = B.height; dev_B.width = B.width, dev_B.stride = B.stride;
    dev_OUT.height = OUT.height; dev_OUT.width = OUT.width, dev_OUT.stride = OUT.stride;

    // Transfer data from host to device memory
    cudaMemcpy(dev_A.elements, A.elements, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B.elements, B.elements, SIZE, cudaMemcpyHostToDevice);

    // Executing kernel 
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((COLS + threads.x - 1) / threads.x, (ROWS + threads.y - 1) / threads.y);
    if (threads.x * blocks.x < ROWS || threads.y * blocks.y < COLS) {
        printf("Program terminated. Block dim: %i, %i, Grid dim: %i, %i, Total threads: %i, %i.\n", threads.x, threads.y, blocks.x, blocks.y, threads.x * blocks.x, threads.y * blocks.y);
        return 0;
    }
    matrix_multi<<<blocks, threads>>>(dev_OUT, dev_A, dev_B);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Runtime API Error reported : %s in file %s on line.\n", cudaGetErrorString(err), __FILE__);
    }
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();  
    
    // Transfer data back to host memory
    cudaMemcpy(OUT.elements, dev_OUT.elements, SIZE, cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix A\n");
    print_mat(A);
    printf("\n\n");
    printf("Matrix OUT\n");
    print_mat(OUT);

    // Verification
    int count = 0, length = 0, i = 0, j = 0;
    for (i = 0; i < ROWS; i++) {
        for(j = 0; j < COLS; j++){
            //assert(fabs(OUT.elements[ij(i, j)] / A.elements[ij(i, j)] - B.elements[ij(i, j)]) < MAX_ERR);
            //if (fabs(OUT.elements[ij(i, j)] / A.elements[ij(i, j)] - B.elements[ij(i, j)]) > MAX_ERR) {
            if (fabs(OUT.elements[ij(i,j)] - A.width * 6) > MAX_ERR) {
                count++;
            }
            length++;
        }
    }
    printf("Verification: %i elements have failed, total length %i, shape: (%i, %i).\n", count, length, i, j);

    // Deallocate device memory
    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_OUT.elements);
    
    // Deallocate host memory
    free(A.elements); 
    free(B.elements); 
    free(OUT.elements);
    
    // flush profile data
    // cuProfilerStop();
}



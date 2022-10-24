#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "cumat.h"


static __global__
void coalescedMemAccess(float * read, float * write, int width, int height) {
    // Assuming the data is aligned and width * height is multiple of warp size
    // read row wise
    float a = read[threadIdx.x + blockDim.x*blockIdx.x + (threadIdx.y + blockDim.y*blockIdx.y) * width];
    write[threadIdx.x + blockDim.x*blockIdx.x + (threadIdx.y + blockDim.y*blockIdx.y) * width] = a;
}

static __global__
void nonCoalescedMemAccess(float * read, float * write, int width, int height) {
    // Assuming the data is aligned and width * height is multiple of warp size
    // read column wise 
    float a = read[(threadIdx.y + blockDim.y*blockIdx.y) + (threadIdx.x + blockDim.x*blockIdx.x) * height];
    write[(threadIdx.y + blockDim.y*blockIdx.y) + (threadIdx.x + blockDim.x*blockIdx.x) * height] = a;
}


/**
 * @brief Use the profiler nvprof and see whats the read throughput.
 */
int main() {
    int N = 1024 * 2 * 2 * 2 * 2;
    cuCV::CuMat<CUCV_32F> A = cuCV::onesOnDevice<CUCV_32F>(N,N,1);
    cuCV::CuMat<CUCV_32F> B = cuCV::zerosOnDevice<CUCV_32F>(N,N,1);

    // Construct Grid.
    const dim3 threads(16, 16);
    const dim3 blocks((A.getWidth() + threads.x - 1) / threads.x, (A.getHeight() + threads.y - 1) / threads.y, A.getNChannels());

    coalescedMemAccess<<<blocks, threads>>>(A.getDataPtr(), B.getDataPtr(), N, N);
    nonCoalescedMemAccess<<<blocks, threads>>>(A.getDataPtr(), B.getDataPtr(), N, N);
}


// From the profiler nvprof:
//
// sudo nvprof -m dram_read_throughput,dram_read_transactions,dram_write_throughput,dram_write_transactions,achieved_occupancy,dram_utilization,flop_sp_efficiency,flop_count_sp_add,single_precision_fu_utilization ./bin/memaccess 
//
// Kernel: coalescedMemAccess(float*, float*, int, int)
//       1                      dram_read_throughput                     Device Memory Read Throughput  39.863GB/s  39.863GB/s  39.863GB/s
//       1                    dram_read_transactions                   Device Memory Read Transactions     2097166     2097166     2097166
//       1                     dram_write_throughput                    Device Memory Write Throughput  39.892GB/s  39.892GB/s  39.892GB/s
//       1                   dram_write_transactions                  Device Memory Write Transactions     2098685     2098685     2098685
//       1                        achieved_occupancy                                Achieved Occupancy    0.871240    0.871240    0.871240
//       1                          dram_utilization                         Device Memory Utilization    High (8)    High (8)    High (8)
//       1                        flop_sp_efficiency                      FLOP Efficiency(Peak Single)       0.00%       0.00%       0.00%
//       1                         flop_count_sp_add   Floating Point Operations(Single Precision Add)           0           0           0
//       1           single_precision_fu_utilization        Single-Precision Function Unit Utilization     Low (2)     Low (2)     Low (2)
// Kernel: nonCoalescedMemAccess(float*, float*, int, int)
//       1                      dram_read_throughput                     Device Memory Read Throughput  13.456GB/s  13.456GB/s  13.456GB/s
//       1                    dram_read_transactions                   Device Memory Read Transactions     2097186     2097186     2097186
//       1                     dram_write_throughput                    Device Memory Write Throughput  13.482GB/s  13.482GB/s  13.482GB/s
//       1                   dram_write_transactions                  Device Memory Write Transactions     2101173     2101173     2101173
//       1                        achieved_occupancy                                Achieved Occupancy    0.870695    0.870695    0.870695
//       1                          dram_utilization                         Device Memory Utilization     Low (3)     Low (3)     Low (3)
//       1                        flop_sp_efficiency                      FLOP Efficiency(Peak Single)       0.00%       0.00%       0.00%
//       1                         flop_count_sp_add   Floating Point Operations(Single Precision Add)           0           0           0
//       1           single_precision_fu_utilization        Single-Precision Function Unit Utilization     Low (1)     Low (1)     Low (1)
//
// Decreasing Througput is expected but somehow the Device Memory Read Transactions are still the same ?!
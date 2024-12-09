// This program performs sum reduction with an optimization removing warp bank conflicts

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

// For last iteration (saves useless work) [Take advantage of warp-level operations]
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
__device__ void warpReduce(volatile int* shmem_ptr, int t) {
    shmem_ptr[t] += shmem_ptr[t + 32];// Only first 32 threads active
    shmem_ptr[t] += shmem_ptr[t + 16];// Only first 16 threads active
    shmem_ptr[t] += shmem_ptr[t + 8];// Only first 8 threads active
    shmem_ptr[t] += shmem_ptr[t + 4];// Only first 4 threads active
    shmem_ptr[t] += shmem_ptr[t + 2];// Only first 2 threads active
    shmem_ptr[t] += shmem_ptr[t + 1];// Only first thread active
}

// Total elements: 65,536
// Each block handles: 512 elements (because each thread handles 2 elements and we have 256 threads per block)
// Number of blocks needed: 65,536 ÷ 512 = 128 blocks
__global__ void sum_reduction(int *v, int *v_r) {
    //Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    //calculate thread id
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load the data into shared memory and do the first add of reduction
    // Vector is now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // because each block handles 512 elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x]; // First partial sum (each block handles 512 elements of the vector)
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    // Stop early (call device function instead)
    for (int s = blockDim.x / 2; s>32; s>>=1) { // s>>=1 == s=s/2 (bitwise right shift assignment operator is faster than division)
        //The right shift >> literally shifts all bits to the right by 1 position, which for positive numbers is the same as dividing by 2:
        // 128 in binary: 10000000
        // 64  in binary: 01000000  (after >>=1)
        // 32  in binary: 00100000  (after >>=1)
        
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);// Warp = 32 threads
    }

    //let the thread 0 for this block write it's result to main memory
    //result is indexed by this block
    if (threadIdx.x == 0){
        v_r[blockIdx.x] = partial_sum[0];
    }
}

void initialize_vector (int *v, int n) {
    for (int i = 0; i < n; i++){
        v[i] = 1;
    }
}

int main() {
    // Define and allocate
    int n = 1<<16;
    size_t bytes = n * sizeof(int); // Memory size | size_t = unsigned integer type (to represent sizes and counts in bytes)

    int *h_v, *h_v_r;
    int *d_v, *d_v_r;
    h_v = (int*)malloc(bytes);
    h_v_r = (int*)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    initialize_vector(h_v, n);

    // Transfer to GPU
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
   

    int TB_SIZE = SIZE;
    int GRID_SIZE = (int)ceil(n/TB_SIZE/2);

    sum_reduction <<<GRID_SIZE,TB_SIZE>>> (d_v, d_v_r);
    sum_reduction <<<1, TB_SIZE>>> (d_v_r, d_v_r);

    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Total sum: %d \n", h_v_r[0]);
    assert(h_v_r[0] == 65536);
    return 0;
}

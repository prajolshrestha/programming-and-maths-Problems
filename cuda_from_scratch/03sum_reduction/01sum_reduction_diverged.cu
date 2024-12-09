#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define SIZE 256 // Array size (power of 2 for optimal performance)
#define SHMEM_SIZE 256 * 4 // Shared memory size as 1024 bytes | SHMEM is fast on-chip memory (can be accessed by all threads in a block)

__global__ void sum_reduction(int *v, int *v_r){
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Reduction loop: Iterate of log base 2 the block dimension
    for (int s = 1; s < blockDim.x; s *= 2){
        // Reduce the threads performing work by half previous the previous iteration each cycle
        if (threadIdx.x % (s * 2) == 0) { // PROBLEM: Half of the threads are idle!
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let the thread 0 for this block write its result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

void initialize_vector (int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1; // rand() % 10;
    }
}


// nvcc -o sum_reduction_diverged 01sum_reduction_diverged.cu
// ./sum_reduction_diverged
// nvprof ./sum_reduction_diverged
int main() {
    // vector size
    int n = 1 << 16;
    size_t bytes = n * sizeof(int);

    // Original vector and result vector
    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    // Allocate memory
    h_v = (int*)malloc(bytes);
    h_v_r = (int*)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    // Initialize vector
    initialize_vector(h_v, n);

    // copy to device
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    // TB and Grid size
    int TB_SIZE = SIZE;
    int GRID_SIZE = (int)ceil(n / TB_SIZE);

    // call kernel
    sum_reduction <<<GRID_SIZE, TB_SIZE>>> (d_v, d_v_r);
    sum_reduction <<<1, TB_SIZE>>> (d_v_r, d_v_r);

    // copy to host
    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Total sum is: %d \n", h_v_r[0]);
    assert(h_v_r[0] == 65536);
    return 0;
}
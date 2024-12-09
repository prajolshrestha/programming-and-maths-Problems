// This program performs sum reduction with an optimization removing warp divergence

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction (int *v, int *v_r) {
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // Cacluate tid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Reduction loop
    for (int s = 1; s < blockDim.x; s *= 2){
        // Just change indexing to be sequential threads (such that we use all the threads)
        int index = 2 * s * threadIdx.x; //PROBLEM: May encounter bank conflicts during memory access!
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }

    // Final value
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }

}

void initialize_vector (int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1; // rand() % 10;
    }
}

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
// This program performs sum reduction with an optimization removing warp bank conflicts

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction (int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // load elements and do first add of reduction
    // vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Assume we have double nr. of threads!

    // store first partial result insted of just the elements
    partial_sum[threadIdx.x] = v[tid] + v[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // 
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

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
    int GRID_SIZE = (int)ceil(n / TB_SIZE / 2); // cut in half // Saves half thread utilization!

    // call kernel
    sum_reduction <<<GRID_SIZE, TB_SIZE>>> (d_v, d_v_r);
    sum_reduction <<<1, TB_SIZE>>> (d_v_r, d_v_r);

    // copy to host
    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Total sum is: %d \n", h_v_r[0]);
    assert(h_v_r[0] == 65536);
    return 0;
}
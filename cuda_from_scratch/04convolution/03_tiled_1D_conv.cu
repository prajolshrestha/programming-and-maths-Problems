// This program implements tiled 1D conv using shared memory

#include <iostream>
#include <stdlib.h>
#include <cassert>
#include <cstdlib>

#define MASK_LENGTH 7
__constant__ int mask[MASK_LENGTH];


__global__ void convolution_1d(int *array, int *result, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Store all elements needed to compute output in shared memory
    extern __shared__ int s_array[];

    int r = MASK_LENGTH / 2;
    int d = 2 * r; // total number of padded elements
    int n_padded = blockDim.x + d; // size of the padded shared memory array
    int offset = threadIdx.x + blockDim.x; // offset for second set of loads in shared memory
    int g_offset = blockDim.x * blockIdx.x + offset; // Global offset for the array in DRAM

    // Load the lower elements first starting at the halo
    // This ensures divergence only once
    s_array[threadIdx.x] = array[tid];

    // Load the remaining upper elements
    if (offset < n_padded){
        s_array[offset] = array[g_offset];
    }
    __syncthreads();

    int temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++){
        temp += s_array[threadIdx.x + j] * mask[j];
    }
    result[tid] = temp;
}

// updated: We have now zero padded array!
void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}

int main(){
    // Number of elements in result array
    int n = 1 << 20;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Size of the mask in bytes
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    // Radius for padding the array
    int r = MASK_LENGTH / 2;
    int n_p = n + r * 2;

    // Size of the padded array in bytes
    size_t bytes_p = n_p * sizeof(int);

    // Allocate the array (include edge elements)...
    int *h_array = new int[n_p];

    // ... and initialize it
    for (int i = 0; i < n_p; i++) {
        if ((i < r) || (i >= (n + r))) {
        h_array[i] = 0;
        } else {
        h_array[i] = rand() % 100;
        }
    }

    // Allocate the mask and initialize it
    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate space for the result
    int *h_result = new int[n];

    // Allocate space on the device
    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_p);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

    // Copy the mask directly to the symbol
    // This would require 2 API calls with cudaMemcpy
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = (n + THREADS - 1) / THREADS;

    // Amount of space per-block for shared memory
    // This is padded by the overhanging radius on either side
    size_t SHMEM = (THREADS + r * 2) * sizeof(int);

    // Call the kernel
    convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

    // Copy back the result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array, h_mask, h_result, n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    cudaFree(d_result);

    return 0;
}
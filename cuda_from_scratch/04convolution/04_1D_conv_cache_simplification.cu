// This program implements cache simplified 1D conv using shared memory

#include <iostream>
#include <stdlib.h>
#include <assert.h>

#define MASK_LENGTH 7
__constant__ int mask[MASK_LENGTH]; // constant memory


__global__ void convolution_1d(int *array, int *result, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Store all elements needed to compute output in shared memory
    extern __shared__ int s_array[]; // Shared memory

    // Load elements from main array into shared memory
    // This is naturally offset by 'r' due to padding
    s_array[threadIdx.x] = array[tid];

    __syncthreads();
    
    int temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++){
        // Get the array values from cache
        if((threadIdx.x + j) >= blockDim.x){
            temp += array[tid + j] * mask[j];
        // Get the value from shared memory
        // Only the last warp will be diverged (given mask size)
        }else{
            temp += s_array[threadIdx.x + j] * mask[j];
        }
    }
    result[tid] = temp;
}

// updated: We have now zero padded array!
void verify_result(int *array, int *mask, int *result, int n){
    int temp;
    for (int i = 0; i < n; i++){
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++){
            temp += array[i + j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main(){
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    // Radius for padding the array
    int r = MASK_LENGTH / 2;
    int n_p = n + r * 2;
    size_t bytes_p = n_p * sizeof(int); // Size of the padded array in bytes

    // Allocate and initialize array (including edge elements)
    int *h_array = new int[n_p];
    for (int i = 0; i < n_p; i++){
        if ((i < r) || (i >= (n + r))){
            h_array[i] = 0;
        }else{
            h_array[i] = rand() % 100;
        }
    }

    // Allocate and initialize mask
    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++){
        h_mask[i] = rand() % 10;
    }

    // Allocate space for result
    int *h_result = new int[n];

    // Allocate space on device
    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_p);
    cudaMalloc(&d_result, bytes_n);

    // copy data to device
    cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

    // copy mask directly to symbol
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;

    // Amount of space per block for shared memory
    // This is padded by the overhanging radius on either side
    size_t SHMEM = THREADS * sizeof(int);

    // Call kernel
    convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

    // copy back result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify 
    verify_result(h_array, h_mask, h_result, n);

    std::cout<<"Completed successfully\n";

    delete [] h_array;
    delete [] h_result;
    delete [] h_mask;
    cudaFree(d_result);
    cudaFree(d_array);
    return 0;
}
// This program implements a 1D convolution using CUDA and stores the mask in constant memory

#include <iostream>
#include <stdlib.h>
#include <assert.h>

using namespace std;

#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void convolution_1d(int *array, int *result, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int r = MASK_LENGTH / 2;
    int start = tid - r;
    int temp = 0;

    for (int j = 0; j <= MASK_LENGTH; j++){
        if ((start + j >= 0) && (start + j < n)){
            temp += array[start + j] * mask[j];
        }
    }
    result[tid] = temp;
}

// Verify
void verify_result(int *array, int *mask, int *result, int n){
    int radius = MASK_LENGTH / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++){
        start = i - radius;
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++){
            if ((start + j >= 0) && (start + j < n)){
                temp += array[start + j] * mask[j];
            }
        }
        assert(result[i] == temp);
    }
}

int main(){
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    int bytes_m = MASK_LENGTH * sizeof(int);

    // Allocate and initialize array
    int *h_array = new int[n];
    for (int i = 0; i < n; i++){
        h_array[i] = rand() % 100;
    }

    // Allocate and initilaize mask
    int *h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++){
        h_mask[i] = rand() % 9;
    }

    // allocate space for result
    int *h_result = new int[n];

    // allocate space on device
    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

    //Copy data directly to symbol
    // would require 2 API calls with cudaMemcpy
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Call kernel
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;
    convolution_1d <<<GRID, THREADS>>>(d_array, d_result, n);

    // copy back the result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify
    verify_result(h_array, h_mask, h_result, n);
    cout<<"Completed Successfully."<<endl;

    return 0;
}
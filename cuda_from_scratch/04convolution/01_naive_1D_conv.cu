// This program implements a 1D convolution using CUDA

#include <iostream>
#include <stdlib.h>
#include <assert.h>

using namespace std;

// 1D Convolution kernel
__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of mask
    int r = m / 2;

    // Calculate the starting point for the element
    int start = tid - r;

    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j < m; j++){
        // Ignore elements outside the boundary
        if ((start + j >= 0) && (start + j < n)){
            temp += array[start + j] * mask[j];
        }
    }

    result[tid] = temp;
}

void verify_result(int *array, int *mask, int *result, int n, int m){
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++){
        start = i - radius;
        temp = 0;
        for (int j =0; j < m; j++){
            if ((start + j >= 0) && (start + j < n)){
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }

}

int main(){
    // Number of elements in result array
    int n = 1 << 20;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Number of elements in the convolution mask
    int m = 7;

    // Size of mask in bytes 
    int bytes_m = m * sizeof(int);

    // Allocate the array (include edge elements)
    int *h_array = new int[n];

    for (int i = 0; i < n; i++){
        h_array[i] = rand() % 100;
    }

    // Allocate mask and initialize it
    int *h_mask = new int[m];

    for (int j = 0; j < m; j++){
        h_mask[j] = rand() % 10;
    }

    // Allocate space for result
    int *h_result = new int[n];

    // Allocate space on the device
    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

    // Threads
    int THREADS = 256;
    int GRID = (n + THREADS - 1) / THREADS;

    // Call kernel
    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify result
    verify_result(h_array, h_mask, h_result, n, m);
    cout<< "Completed Successfully"<<endl;

    return 0;
}
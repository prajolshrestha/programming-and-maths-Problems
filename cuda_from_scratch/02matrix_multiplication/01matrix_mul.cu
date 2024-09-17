// Matrix multiplication
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h> 

__global__ void matrixMul(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; //Compute each thread's row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; //Compute each thread's column index

    int temp_sum = 0;
    if ( (row < n) && (col < n)) { // Boundary production
        // Iterate over row and down column
        for (int k = 0; k < n; k++) {
            // Accumulate result for single element
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = temp_sum;
    }
}

void init_matrices(int *a, int *b, int n) {
    for (int i =0; i < n; i++){
        for (int j = 0; i < n; i++){
            a[i * n + j] = rand() % 100;
            b[i * n + j] = rand() % 100;    
        }
        
    }
}

void verify_result(int*a, int *b, int *c, int n){
    int *verify_c;
    verify_c = (int*)malloc(n * n * sizeof(int));
    
    for (int i=0; i < n; i++){
        for (int j = 0; j < n; j++){
            for (int k=0; k < n; k++){
                verify_c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            assert(c[i * n + j] == verify_c[i * n + j]);
            }
    }

}

// nvcc -o matrix_mul 01matrix_mul.cu

int main() {
    // Matrix of size 1024 x 1024
    int n = 1 << 10;
    // size (in bytes) of the matrix
    size_t bytes = n * n * sizeof(int);

    // CPU
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    init_matrices(h_a, h_b, n);

    // GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define variables of thread
    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Lunch kernel
    matrixMul <<<grid, threads>>> (d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // check result
    verify_result(h_a, h_b, h_c, n);

    printf("Completed successfully");

    return 0;


}

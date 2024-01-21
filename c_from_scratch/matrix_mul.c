#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void matrix_mul(float *a, float *b, float *c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (row < size && col < size) {
        for (int i = 0; i < size; i++) {
            sum += a[row*size + i] * b[i*size + col];
        }
        c[row*size + col] = sum;
    }
}

int main() {
    int size = 128;
    int n = size*size;

    float *a, *b, *c;
    float *a_gpu, *b_gpu, *c_gpu;

    a = (float*)malloc(n*sizeof(float));
    b = (float*)malloc(n*sizeof(float));
    c = (float*)malloc(n*sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = (float)rand()/(float)RAND_MAX;
        b[i] = (float)rand()/(float)RAND_MAX;
    }

    cudaMalloc((void**)&a_gpu, n*sizeof(float));
    cudaMalloc((void**)&b_gpu, n*sizeof(float));
    cudaMalloc((void**)&c_gpu, n*sizeof(float));

    cudaMemcpy(a_gpu, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((size + BLOCK_SIZE - 1)/BLOCK_SIZE, (size + BLOCK_SIZE - 1)/BLOCK_SIZE);

    matrix_mul<<<grid_size, block_size>>>(a_gpu, b_gpu, c_gpu, size);

    cudaMemcpy(c, c_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input matrix A:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", a[i*size + j]);
        }
        printf("\n");
    }

    printf("\nInput matrix B:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", b[i*size + j]);
        }
        printf("\n");
    }

    printf("\nOutput matrix C:\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", c[i*size + j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return 0;
}

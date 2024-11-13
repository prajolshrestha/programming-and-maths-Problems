#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>


// Initiate a vector
void vector_init(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 100);
    }
}

// verify the result
void verify_result (float *a, float *b, float *c, float factor, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == factor * a [i] + b[i]);
    }
}


// nvcc -o cublas_vecadd 05cuBLAS_vec_add.cu -lcublas -lcurand
int main() {
    int n = 1 << 2;
    size_t bytes = n * sizeof(float);

    // Declare vector pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    // Allocate memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);


    // Initialize vectors
    vector_init(h_a, n);
    vector_init(h_b, n);

    // create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // copy the vectors over to the device
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    // Lunch simple saxpy kernel (single precision a * x + y)
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

    // copy the result vector back out
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    // Print out the result
    verify_result(h_a, h_b, h_c, scale, n);

    // clearn up the created handle
    cublasDestroy(handle);

    // Release allocated memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    printf("Completed Successfully\n");

    return 0;

}
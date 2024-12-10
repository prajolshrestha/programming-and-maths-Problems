// This program implements 2D convolution using constant memory in CUDA

#include <cassert>
#include <cstdlib>
#include <iostream>

# define MASK_DIM 7
# define MASK_OFFSET (MASK_DIM / 2)
__constant__ int mask[7 * 7];

__global__ void convolution_2d(int *matrix, int *result, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    int temp = 0;

    for (int i = 0; i < MASK_DIM; i++){
        for (int j = 0; j < MASK_DIM; j++){
            if ((start_r + i >= 0) && (start_r + i < n)){
                if ((start_c + j >= 0) && (start_c + j < n)){
                    temp += matrix[(start_r + i) * n + (start_c + j)] * mask[i * MASK_DIM + j];
                }
            }
        }
    }
    result[row * n + col] = temp;
}

void init_matrix(int *matrix, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            matrix[n * i + j] = rand() % 100;
        }
    }
}

void verify_result(int *m, int *mask, int *result, int n){
    int temp;

    // Intermediate value for more readable code
    int offset_r;
    int offset_c;

    
    for (int i = 0; i < n; i++){ // Go over each row
        for (int j = 0; j < n; j++){ // Go over each col
            temp = 0; // reset

            for (int k = 0; k < MASK_DIM; k++){
                offset_r = i - MASK_OFFSET + k;
                for (int l = 0; l < MASK_DIM; l++){
                    offset_c = j - MASK_OFFSET + l;

                    // Range check
                    if (offset_r >= 0 && offset_r < n){
                        if (offset_c >= 0 && offset_c < n){
                            temp += m[offset_r * n + offset_c] * mask[k * MASK_DIM + l];
                        }
                    }
                }
            }
            assert(temp == result[i * n + j]);
        }
    }
}

int main(){
    int n = 1 << 10;
    size_t bytes_n = n * n * sizeof(int);
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);


    // Allocate and initialize matrix
    int *matrix = new int[n*n];
    int *result = new int[n*n];
    init_matrix(matrix, n);

    // Allocate and initilalize mask
    int *h_mask = new int[MASK_DIM * MASK_DIM];
    init_matrix(h_mask, MASK_DIM);

    // Allocate space on device
    int *d_matrix; 
    int *d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // copy data to device
    cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    int THREADS = 16;
    int BLOCKS = (n + THREADS - 1) / THREADS;

    // dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    // call the kernel
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, n);
    cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);
    verify_result(matrix, h_mask, result, n);

    std::cout<<"Completed sucessfully!\n";

    delete [] matrix;
    delete [] result;
    delete [] h_mask;
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
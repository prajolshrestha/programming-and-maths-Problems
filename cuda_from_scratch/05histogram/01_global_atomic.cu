// This program implements histogram in cuda
#include <iostream>
#include <cassert>
#include <fstream>
// #include <algorithm>
// #include <cstdlib>
// #include <numeric>



// #define BINS 7
// #define DIV ((26 + BINS - 1) / BINS)

// constexpr is type-safe; the compiler knows these are integers
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS); 


__global__ void histogram(char *a, int *result, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the bin positions where threads are grouped together
    int alpha_position;
    for (int i = tid; i < n; i += (gridDim.x * blockDim.x)){
        // calculate the postion in the alphabet
        alpha_position = a[i] - 'a';
        atomicAdd(&result[alpha_position / DIV], 1);
    }


}


void init_array(char *a, int n){
    srand(1);
    for(int i = 0; i < n; i++){
        a[i] = 'a' + rand() % 26;
    }
}


int main(){
    // Declare our problem size
    int n = 1 << 20;
    size_t bytes_n = n * sizeof(char);

    // Allocate memory on the host
    char *h_a = new char[n];

    // Allocate space for binned results
    int *h_result = new int[BINS]();
    size_t bytes_r = BINS * sizeof(int);

    // Initialize the array
    init_array(h_a, n);

    // Allocate memory on device
    char *d_a;
    int *d_result;
    cudaMalloc(&d_a, bytes_n);
    cudaMalloc(&d_result, bytes_r);

    // Copy the array to device
    cudaMemcpy(d_a, h_a, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, bytes_r, cudaMemcpyHostToDevice);

    // Number of threads per threadblock
    int THREADS = 512;
    int BLOCKS = n / THREADS;
    histogram<<<BLOCKS, THREADS>>>(d_a, d_result, n);

    cudaMemcpy(h_result, d_result, bytes_r, cudaMemcpyDeviceToHost);

    // Write the data out for gnuplot
    std::ofstream output_file;
    output_file.open("global_mem_histogram.dat", std::ios::out | std::ios::trunc);

    for (int i = 0; i < BINS; i++){
        output_file << h_result[i] << "\n\n";
    }
    output_file.close();

    std::cout<<"Completed Successfully!\n";

    return 0;
}
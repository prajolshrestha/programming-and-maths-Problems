// Vector addition (pinned_memory)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>

using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::generate;
using std::vector;

// Cuda kernel for vector addition
__global__ void vecAdd(int* a, int* b, int* c, int N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}

}

// check result
void verify_result(int* a, int* b, int* c, int N) {
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}

}





int main() {

	// Initialize a vector with random numbers
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;
	
	// Allocate memory on CPU
	int* h_a, * h_b, * h_c;
	cudaMallocHost(&h_a, bytes);
	cudaMallocHost(&h_b, bytes);
	cudaMallocHost(&h_c, bytes);

	// initialize
	for (int i = 0; i < N; i++) {
		h_a[i] = rand() % 100;
		h_b[i] = rand() % 100;
	}

	// Allocate memory on GPU
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data from host to Device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// THREAD
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	vecAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Verify calculation
	verify_result(h_a, h_b, h_c, N);

	// Free Memory
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout << "Completed Successfully.\n";

	return 0;

}
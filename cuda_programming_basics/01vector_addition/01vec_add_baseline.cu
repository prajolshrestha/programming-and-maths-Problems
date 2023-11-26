// This program computes the sum of two vectors of length N (baseline)

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>


using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::endl;
using std::vector;
using std::generate;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int N) { // __global__ means this is called from CPU and will run on the GPU

	// Calculate global thread ID
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Boundary check
	if (tid < N) {
		// Each thread adds a single element
		c[tid] = a[tid] + b[tid];
	}
}

// Check vector add result 
void verify_result(vector<int> a, vector<int> b, vector<int> c) {
	for (int i = 0; i < a.size(); i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {

	/// 1. Initialize vectors with random numbers (on CPU)
	// Array size of 2^16 (65536 elements)
	constexpr int N = 1 << 16;
	size_t bytes = sizeof(int) * N; // used later for allocation

	// Vector for holding the host-side (cpu-side) data
	vector<int> a(N);
	vector<int> b(N);
	vector<int> c(N);

	// Initialize random numbers in each array
	generate(begin(a), end(a), []() {return rand() % 100; }); // every element is in between 0 and 99 inclusive
	generate(begin(b), end(b), []() {return rand() % 100; });

	//// 2. Memory allocation (on GPU)
	// Allocate memory on the device (GPU)
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes); // d_a le point gareko address bata 'bytes' memory allocate garne
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes); // we do not initialize d_c, as it will certainly be overwritten in the future

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice); // copy 'bytes' data from 'a.data()'(CPU) to 'd_a'(GPU) in the direction 'cudaMemcpyHostToDevice'
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA (1024 threads per CTA)
	int NUM_THREADS = 1 << 10;

	// CTAs per GRID
	// We need to lunch at LEAST as many threads as we have elements.
	// This equation pads an extra CTA to the grid if N cannot evenly be divided by NUM_THREADS
	// eg. N = 1025, NUM_THREADS = 1024 then, NUM_BLOCKS = 1025 / 1024 = 1 , But one thread is missing.
	//  so, Formula becomes NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// Lunch the kernel on the GPU
	// Kernel call are aysnchronous (the CPU program continues execution after call, but no necessarily before the kernel finishes)
	vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, N);

	// Copy sum vector from device to host
	// cudaMemcpy is a synchronous operation, and waits for the prior kernel lunch to complete (both goes to default stream in this case).
	// Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	// Check the results for errors
	verify_result(a, b, c);

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout << "Completed Sucessfully" << endl;

	return 0;


}
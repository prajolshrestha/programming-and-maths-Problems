#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <vector>
#include <iterator>
#include <cstdlib>

__global__ void vecAdd(int* a, int* b, int* c, int N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}

}

void init_vector(int* a, int* b, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;

	}
}

void verify_results(int* a, int* b, int* c, int N) {
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}


int main() {

	constexpr int N = 1 << 16;
	size_t bytes = sizeof(int) * N;

	// Method 0: Baseline
	
	// CPU
	std::vector<int> g(N);
	std::vector<int> h(N);
	std::vector<int> i(N);
	
	//initialize
	std::generate(std::begin(g), std::end(g), []() {return std::rand() % 100; });
	std::generate(std::begin(h), std::end(h), []() {return std::rand() % 100; });

	// GPU
	int* d_g, * d_h, * d_i;
	cudaMalloc(&d_g, bytes);
	cudaMalloc(&d_h, bytes);
	cudaMalloc(&d_i, bytes);

	cudaMemcpy(d_g, g.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h.data(), bytes, cudaMemcpyHostToDevice);

	// Threads
	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	vecAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_g, d_h, d_i, N);

	cudaMemcpy(i.data(), d_i, bytes, cudaMemcpyDeviceToDevice);

	

	cudaFree(d_g);
	cudaFree(d_h);
	cudaFree(d_i);




	// Method 1: Pinned Memory ///////////////////////////////////////////////
	
	// CPU
	int* a, * b, * c;
	cudaMallocHost(&a, bytes);
	cudaMallocHost(&b, bytes);
	cudaMallocHost(&c, bytes);

	init_vector(a, b, N); // initialize

	// GPU
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	// Work on threads
	//int NUM_THREADS = 1 << 10;
	//int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	vecAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, N); // Kernel call

	// CPU
	cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

	verify_results(a, b, c, N);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	// Method 2:Unified Memory (prefetch) /////////////////////////////////////////////////////

	int id = cudaGetDevice(&id); // get device id for cuda calls

	// CPU
	int* d, * e, *f;
	cudaMallocManaged(&d, bytes);
	cudaMallocManaged(&e, bytes);
	cudaMallocManaged(&f, bytes);
	
	init_vector(d, e, N);

	// GPU [Transfer not required, done automatically]
	// Thread
	int BLOCK_SIZE = 1 << 10;
	int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

	vecAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d, e, f, N);

	// prefetch for performance boost
	cudaMemPrefetchAsync(d, bytes, id);
	cudaMemPrefetchAsync(e, bytes, id);

	cudaDeviceSynchronize(); // synchronize 

	cudaMemPrefetchAsync(f, bytes, cudaCpuDeviceId); // prefetch

	verify_results(d, e, f, N);


	
	return 0;



}
// Vector addition using unified memory with prefetch[virtual memory]
// Instead of having seperate device memory and host memory having to manage both these independently,
// Some other system takes care of it. All I worry about is my kernel code. 
// ie, we dont have to manage memory

// To increase performance we use prefetch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


__global__ void vecAddUM(int* a, int* b, int* c, int N) {
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

void check_answer(int* a, int* b, int* c, int N) {
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}


int main() {
	// get the device Id for other CUDA calls
	int id = cudaGetDevice(&id);

	int N = 1 << 16;
	size_t bytes = sizeof(int) * N;

	// Declare unified memory pointers and initialize them with random numbers
	int* a, * b, * c;
	// allocate memory
	cudaMallocManaged(&a, bytes); // Transfer of data betn CPU and GPU happen automatically
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);
	init_vector(a, b, N);

	// Thread
	int BLOCK_SIZE = 256;
	int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

	// call cuda Kernel
	// Performance boost:
	// Problem: GPU starts up and says I dont have any of the data. Page Fault ( page in memory from CPU to GPU) [page size: CPU is of 4kilobytes]
	// When data should be where? ==> Prefetching : Behind the scene you can transfer data while you do other things. (ahead of time)
	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);
	vecAddUM << <GRID_SIZE, BLOCK_SIZE >> > (a, b, c, N);

	// wait for all previous operations before using values
	cudaDeviceSynchronize();

	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId); // Prefetching device memory back to the CPU

	// check result
	check_answer(a, b, c, N);

	printf("COmpleted Suffessfully\n");

	return 0;

}
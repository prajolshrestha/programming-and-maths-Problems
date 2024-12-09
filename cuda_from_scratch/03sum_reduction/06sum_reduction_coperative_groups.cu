// This programs does sum reduction with an optimization removing bank conflicts

// Visual flow:
// [Global Memory: Many Elements]
//          ↓ (Thread-Level Sum)
// [Thread Sums: 256 values per block]
//          ↓ (Block-Level Reduction)
// [Block Sums: 1 value per block]
//          ↓ (Atomic Addition)
// [Final Single Sum]

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int *temp, int val) {
    int lane = g.thread_rank(); // Gets thread's position within the group (eg, block, warp)

    // Reduction loop
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2) {
        temp[lane] = val; // Store current value
        // wait for all threads to store
        g.sync();
        if (lane < i) { // only active threads
            val += temp[lane + i]; // add pair values
        }
        // wait for all threads to load
        g.sync();
    }
    // only thread 0 will return full sum
    return val;
}

__device__ int thread_sum(int *input, int n){
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n/4; i += blockDim.x * gridDim.x) {
        // cast as int4 [int4 is cuda vector type that holds 4 integers]
        int4 in = ((int4*)input)[i]; // loads 4 integrs at once | more efficient
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}


__global__ void sum_reduction (int *sum, int *input, int n) {
    // 1. Each thread sums its chunk of data (4 elements at a time)
    int my_sum = thread_sum(input, n);
    
    // 2. Block-level reduction using shared memory
    extern __shared__ int temp[];
    auto g = this_thread_block();
    int block_sum = reduce_sum(g, temp, my_sum);

    // 3. Combine block results using atomic operation
    if (g.thread_rank() == 0) {
        atomicAdd(sum, block_sum);
    }
}

void initialize_vector (int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1;
    }
}

int main() {
    int n = 1<<13;
    size_t bytes = n * sizeof(int);

    // original vector and result vector
    int *sum;
    int *data;

    // Allocate using unified memory
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&data, bytes);

    initialize_vector(data, n);

    int TB_SIZE = 256;
    int GRID_SIZE = (n + TB_SIZE -1) / TB_SIZE;
    
    // Kernel(number of blocks, nr of threads in a block, shared memory size)
    sum_reduction <<<GRID_SIZE, TB_SIZE, n * sizeof(int) >>> (sum, data, n);

    cudaDeviceSynchronize(); // sync kernel

    // Print the result
    printf("Sum is: %d \n", sum[0]);
    //assert(sum[0] == 65536);

    return 0;
}
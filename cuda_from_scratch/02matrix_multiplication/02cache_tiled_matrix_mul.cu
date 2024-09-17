
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <math.h>

// Static shmem calculation for convenience (int 16 x 16 matrix)
#define SHMEM_SIZE 16 * 16 * 4

// Kernel
__global__ volid tiledMatrixMul(int *a, int *b, int *c, int n, int tile_size) {
     

}


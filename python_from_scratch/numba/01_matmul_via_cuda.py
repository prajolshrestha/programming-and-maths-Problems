'''
    GPU Programming:

    CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.

    Numba is an open source, NumPy-aware optimizing compiler for Python sponsored by Anaconda, Inc. 
    It uses the LLVM compiler project to generate machine code from Python syntax.

    
'''
import numpy as np
from numba import cuda, float32, float64, int32
import cupy as cp

print(cuda.detect())

TPB = 16

@cuda.jit
def matmul(A,B,C):

    ## Define an array in shared memory
    sA = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB), dtype=float32)

    # Absolute index of threads
    x,y = cuda.grid(2) 
    # Index relative to block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x # blocks per grid

    if x>= C.shape[0] and y>= C.shape[1]:
        return #quit, outside the valid memory
    
    tmp = 0.
    for i in range(bpg):
        ## Load data to shared memory
        sA[tx, ty] = A[x, ty+i*TPB]
        sB[tx, ty] = B[tx+i*TPB, y]

        # Syncronize thread preloading
        cuda.syncthreads()

        ## compute partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        
        # wait until all threads finish computing
        cuda.syncthreads()
    
    C[x,y] = tmp

    

SIZE = 4000
A = cp.random.uniform(1,10,size=(SIZE,SIZE), dtype=np.float32)
B = cp.random.uniform(1,10,size=(SIZE,SIZE), dtype=np.float32)
C = cp.zeros((SIZE,SIZE), dtype=np.float32)

threadsperblock = (TPB, TPB)
blockspergrid = int(np.ceil(SIZE/ threadsperblock[0]))
blockspergrid = (blockspergrid, blockspergrid)

C = matmul[blockspergrid, threadsperblock](A,B,C)

### Shortcut
import time
tic = time.time()
D = cp.dot(A,B,C)
toc = time.time()

print(D)
print(toc-tic)
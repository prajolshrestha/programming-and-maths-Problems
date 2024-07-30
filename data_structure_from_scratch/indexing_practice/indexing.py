import numpy as np

matrix = np.linspace(0,24,25).reshape(5,5)

print(matrix.shape)
n = 5
mul = np.zeros((n,n))
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        for k in range(matrix.shape[1]):
            mul[i,j] += matrix[i,k] * matrix[k,j] 
        

print(mul)

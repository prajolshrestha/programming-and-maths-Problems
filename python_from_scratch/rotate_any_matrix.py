import numpy as np
import math

def rotate_any_matrix(matrix, theta):
    # Generate rotation matrix
    radian = math.radians(theta)
    costheta = math.cos(radian)
    sintheta = math.sin(radian)
    rotation_matrix = np.array([[costheta, sintheta],
                                [-sintheta, costheta]])
    print(rotation_matrix)

    # compute center vertices
    h, w = matrix.shape[0], matrix.shape[1]
    center_y, center_x = h / 2, w / 2
    print(center_y, center_x)

    rotated_matrix = np.zeros_like(matrix)
    for i in range(h):
        for j in range(w):
            # Translate coordinates relative to center of the matrix
            y = i - center_y
            x = j - center_x
            
            # Apply rotation matrix
            new_x, new_y = np.dot(rotation_matrix, [x, y])

            # Translate coordinates back to the original frame of reference
            new_i = int(round(new_y + center_y)) # issue
            new_j = int(round(new_x + center_x))

            if 0 <= new_i < h and 0 <= new_j < w:
                rotated_matrix[new_i,new_j] = matrix[i,j] # append value of old matrix to calculated new coordinates
    
    return rotated_matrix

import scipy.ndimage
def rotate_matrix_effectively(matrix, theta):
    return scipy.ndimage.rotate(matrix, theta, reshape=False)


if __name__ == '__main__':
    matrix = np.array([[1,2],
                        [3,4]])
    matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
    theta = 90
    new_matrix = rotate_any_matrix(matrix, theta)
    print(new_matrix)
    mat = rotate_matrix_effectively(matrix, theta)
    print(mat)




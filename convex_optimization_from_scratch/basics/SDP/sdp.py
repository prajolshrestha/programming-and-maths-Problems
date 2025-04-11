"""
This code demonstrates a basic Semidefinite Programming (SDP) problem:

- We create a 3x3 symmetric matrix variable X that must be positive semidefinite
- We minimize trace(C*X) where C is a random symmetric cost matrix 
- Subject to constraints that X is PSD and has trace = 1
- The solution X.value gives us the optimal matrix
- We verify the result by checking eigenvalues are non-negative
- Finally we visualize the results with eigenvalue plots and matrix heatmaps

Matrix Definiteness Types:
+-------------------------+---------------------------+--------------------------------+--------------------------------+
| Type                    | Eigenvalue Condition      | Quadratic Form Condition      | Geometric Interpretation        |
+-------------------------+---------------------------+--------------------------------+--------------------------------+
| Positive Definite (PD)  | All λ > 0                | x^T A x > 0 for x ≠ 0         | Strictly convex ellipsoid      |
| Positive Semidefinite   | All λ ≥ 0                | x^T A x ≥ 0 for all x         | Possibly degenerate ellipsoid  |
| (PSD)                   |                          |                                | (may have flat directions)     |
| Strictly/Strongly       | All λ ≥ c > 0            | x^T A x ≥ c||x||^2            | Strictly convex ellipsoid with |
| Definite (SD)           | for some constant c      | for some c > 0                | minimum axis length            |
+-------------------------+---------------------------+--------------------------------+--------------------------------+

Geometric Interpretation:
- The set of positive semidefinite matrices forms a convex cone in matrix space
- The trace=1 constraint intersects this cone to form a convex set
- Geometrically, a PSD matrix X can be viewed as defining an ellipsoid via x^T X x ≤ 1
- The eigenvalues determine the lengths of the ellipsoid's axes (shape)
- The eigenvectors determine the orientation of these axes (orientation)
- The trace constraint normalizes the "size" of this ellipsoid

Eigenvalue Patterns and Their Geometric Meaning:
+------------------------+----------------------------------------+--------------------------------+
| Eigenvalue Pattern     | Geometric Interpretation               | Matrix Properties              |
+------------------------+----------------------------------------+--------------------------------+
| One λ = 0, others > 0  | Ellipsoid collapsed in one direction  | Rank deficient (rank = 2)      |
| Two λ = 0, one > 0     | Ellipsoid collapsed to a line         | Rank = 1                       |
| All λ = 0             | Ellipsoid collapsed to a point         | Zero matrix                    |
| All λ equal and > 0   | Perfect sphere                         | Multiple of identity matrix    |
| All λ > 0, different  | General ellipsoid                      | Full rank, strictly positive   |
+------------------------+----------------------------------------+--------------------------------+

Objective and Constraints:
- The objective trace(C*X) is a linear function in the space of matrices
- It can be interpreted as the sum of elementwise products of C and X
- The PSD constraint X >> 0 ensures all eigenvalues are non-negative
- This creates a curved boundary in matrix space
- The trace=1 constraint forms a hyperplane
- The optimal solution lies at the intersection of these geometric objects

Geometric Shapes Based on Eigenvalues:
+------------------------+----------------------------------------+--------------------------------+
| Shape                  | Eigenvalue Pattern                     | Example Matrix                 |
+------------------------+----------------------------------------+--------------------------------+
| Ball/Sphere           | All λ equal and > 0                    | Identity matrix scaled by 1/n  |
|                      | e.g. λ₁ = λ₂ = λ₃ = 1/3                | [[1/3  0    0  ]              |
|                      |   Isotropic scaling                     |  [0    1/3  0  ]              |
|                      |                                         |  [0    0    1/3]]             |
+------------------------+----------------------------------------+--------------------------------+
| Ellipsoid            | All λ > 0 but different                | Any PD matrix with trace=1     |
|                      | e.g. λ₁ = 0.5, λ₂ = 0.3, λ₃ = 0.2      | [[0.5  0    0  ]              |
|                      |                                         |  [0    0.3  0  ]              |
|                      |                                         |  [0    0    0.2]]             |
+------------------------+----------------------------------------+--------------------------------+
| Degenerate Ellipsoid | One λ = 0, others > 0                  | Rank-2 matrix with trace=1     |
| (Flat in 1 direction)| e.g. λ₁ = 0.6, λ₂ = 0.4, λ₃ = 0       | [[0.6  0    0  ]              |
|                      |                                         |  [0    0.4  0  ]              |
|                      |                                         |  [0    0    0  ]]             |
+------------------------+----------------------------------------+--------------------------------+
| Line                 | Two λ = 0, one > 0                     | Rank-1 matrix with trace=1     |
|                      | e.g. λ₁ = 1, λ₂ = 0, λ₃ = 0           | [[1.0  0    0  ]              |
|                      |                                         |  [0    0    0  ]              |
|                      |                                         |  [0    0    0  ]]             |
+------------------------+----------------------------------------+--------------------------------+
| Point                | All λ = 0                              | Zero matrix                    |
|                      | e.g. λ₁ = λ₂ = λ₃ = 0                  | [[0    0    0  ]              |
|                      |                                         |  [0    0    0  ]              |
|                      |                                         |  [0    0    0  ]]             |
+------------------------+----------------------------------------+--------------------------------+



SDPs are important in optimization as they can model many practical problems
while still being solvable efficiently.
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

# Problem dimensions
n = 3  # Size of our matrix

# Create a variable for our semidefinite matrix
X = cp.Variable((n, n), symmetric=True)

# Create a random cost matrix
np.random.seed(1)
C = np.random.randn(n, n)
C = (C + C.T) / 2  # Make it symmetric

# Linear objective function
objective = cp.Minimize(cp.trace(C @ X))

# Constraints
constraints = [
    X >> 0,  # This means X is positive semidefinite
    cp.trace(X) == 1  # Normalization constraint
]

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.SCS)

# Display results
print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal X:\n", X.value)

# Verify that X is indeed positive semidefinite
eigenvalues, eigenvectors = np.linalg.eigh(X.value)
print("Eigenvalues of X:", eigenvalues)
print("Is X positive semidefinite?", all(eigenvalues >= -1e-6))  # Allow tiny numerical errors

# Visualization
plt.figure(figsize=(15, 10))  # Increased figure height for 2 rows

# 1. Eigenvalue visualization
plt.subplot(2, 2, 1)
plt.bar(range(n), eigenvalues, color='blue', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Optimal X')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. Matrix heatmap
plt.subplot(2, 2, 2)
im = plt.imshow(X.value, cmap='viridis')
plt.colorbar(im, label='Value')
plt.title('Heatmap of Optimal X')
for i in range(n):
    for j in range(n):
        plt.text(j, i, f'{X.value[i, j]:.2f}', ha='center', va='center', color='white')

# 3. 2D Ellipsoid visualization
if n >= 2:
    plt.subplot(2, 2, 3)
    
    # We'll plot a 2D slice if n > 2
    if n > 2:
        # Use the top 2x2 submatrix for visualization
        vis_matrix = X.value[:2, :2]
        plt.title('2D Slice of Ellipsoid Represented by X')
    else:
        vis_matrix = X.value
        plt.title('Ellipsoid Represented by X')
    
    # Calculate eigenvalues and eigenvectors for the 2D visualization
    evals_2d, evecs_2d = np.linalg.eigh(vis_matrix)
    
    # Create points for a unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    
    # Transform the circle to our ellipse using eigenvectors and square root of eigenvalues
    ellipse = np.zeros((2, 100))
    for i in range(2):
        ellipse += np.outer(evecs_2d[:, i], 1.0/np.sqrt(max(evals_2d[i], 1e-10)) * np.array([np.cos(theta), np.sin(theta)])[i])
    
    plt.plot(ellipse[0, :], ellipse[1, :], 'r-')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X₁')
    plt.ylabel('X₂')

# 4. 3D Ellipsoid visualization (if n >= 3)
if n >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    
    ax = plt.subplot(2, 2, 4, projection='3d')
    
    # Use the top 3x3 submatrix (or the full matrix if n=3)
    vis_matrix_3d = X.value[:3, :3]
    
    # Calculate eigenvalues and eigenvectors for the 3D visualization
    evals_3d, evecs_3d = np.linalg.eigh(vis_matrix_3d)
    
    # Create a meshgrid for a unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Reshape to stack the coordinates
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    
    # Transform the sphere to our ellipsoid
    transformed_points = np.zeros_like(points)
    for i in range(3):
        # Scale by the inverse square root of eigenvalues (multiply for ellipsoid transformation)
        # The 1/sqrt(eigenvalue) gives the correct scaling for the ellipsoid
        transformed_points += np.outer(evecs_3d[:, i], 1.0/np.sqrt(max(evals_3d[i], 1e-10)) * points[i, :])
    
    # Reshape back to grid for plotting
    x_ellipsoid = transformed_points[0, :].reshape(x.shape)
    y_ellipsoid = transformed_points[1, :].reshape(y.shape)
    z_ellipsoid = transformed_points[2, :].reshape(z.shape)
    
    # Plot the ellipsoid
    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, 
                    rstride=1, cstride=1, color='r', alpha=0.6, linewidth=0)
    
    # Set labels and title
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_zlabel('X₃')
    ax.set_title('3D Ellipsoid Represented by X')
    
    # Make the plot more visually appealing
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

plt.tight_layout()
#plt.savefig('sdp_visualization.png', dpi=300)
plt.show()

print("Visualization includes 2D and 3D representations of the ellipsoid")

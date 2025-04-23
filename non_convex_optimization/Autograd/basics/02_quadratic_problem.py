import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # Define a quadratic function: f(x) = x^T A x
    # This is the form of a quadratic function where:
    # - x is a vector
    # - A is a symmetric matrix
    # - The result is a scalar
    
    # Define matrix A (symmetric for a valid quadratic form)
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    
    def quadratic_function(x):
        """Compute the quadratic form x^T A x"""
        return np.dot(x, np.dot(A, x))
    
    # Compute the gradient function
    grad_quadratic = grad(quadratic_function)
    
    # Evaluate at a point
    x_point = np.array([1.0, 2.0])
    function_value = quadratic_function(x_point)
    gradient_value = grad_quadratic(x_point)
    
    print(f"Matrix A:\n{A}")
    print(f"\nPoint x: {x_point}")
    print(f"f(x) = x^T A x = {function_value}")
    print(f"∇f(x) = {gradient_value}")
    
    # True gradient should be 2Ax
    true_gradient = 2 * np.dot(A, x_point)
    print(f"Analytical gradient 2Ax: {true_gradient}")
    
    # Create a grid for visualization
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # Compute function values for each grid point
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            point = np.array([X[j, i], Y[j, i]])
            Z[j, i] = quadratic_function(point)
    
    # Create gradient field for visualization
    X_grad = np.linspace(-3, 3, 10)
    Y_grad = np.linspace(-3, 3, 10)
    X_grad_mesh, Y_grad_mesh = np.meshgrid(X_grad, Y_grad)
    U = np.zeros_like(X_grad_mesh)
    V = np.zeros_like(Y_grad_mesh)
    
    # Compute gradients at each point
    for i in range(len(X_grad)):
        for j in range(len(Y_grad)):
            point = np.array([X_grad_mesh[j, i], Y_grad_mesh[j, i]])
            grad_value = grad_quadratic(point)
            U[j, i] = grad_value[0]
            V[j, i] = grad_value[1]
    
    # Implement gradient descent to find minimum
    print("\nGradient Descent to find minimum:")
    
    # Initialize
    x = np.array([2.0, 2.0])  # Starting point
    learning_rate = 0.1
    iterations = 10
    trajectory = [x.copy()]
    function_values = [quadratic_function(x)]
    
    for i in range(iterations):
        # Compute gradient
        grad_value = grad_quadratic(x)
        
        # Update x
        x = x - learning_rate * grad_value
        
        # Store current position and function value
        trajectory.append(x.copy())
        function_values.append(quadratic_function(x))
        
        print(f"Iteration {i+1}: x = {x}, f(x) = {quadratic_function(x)}")
    
    trajectory = np.array(trajectory)
    
    # Create a single figure with 6 subplots (3x2 grid)
    plt.figure(figsize=(20, 15))
    
    # 1. 2D Contour Plot of the Function
    plt.subplot(3, 2, 1)
    contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar(contour, label='f(x) = x^T A x')
    plt.contour(X, Y, Z, 20, colors='white', alpha=0.5, linestyles='solid', linewidths=0.5)
    plt.title('2D: Quadratic Function f(x) = x^T A x')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    
    # 2. 3D Surface Plot of the Function
    ax3d = plt.subplot(3, 2, 2, projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax3d.set_title('3D: Quadratic Function f(x) = x^T A x')
    ax3d.set_xlabel('x₁')
    ax3d.set_ylabel('x₂')
    ax3d.set_zlabel('f(x)')
    plt.colorbar(surf, ax=ax3d, shrink=0.5, label='Function value')
    
    # 3. 2D Gradient Field
    plt.subplot(3, 2, 3)
    plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.3)
    plt.quiver(X_grad_mesh, Y_grad_mesh, U, V, color='red', scale=50)
    plt.scatter(x_point[0], x_point[1], color='blue', s=100, marker='o')
    plt.arrow(x_point[0], x_point[1], 
              gradient_value[0]/3, gradient_value[1]/3,  # Scaled for visibility
              head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    plt.title('2D: Gradient Field ∇f(x)')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    
    # 4. 3D Gradient Visualization
    ax3d2 = plt.subplot(3, 2, 4, projection='3d')
    ax3d2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')
    
    # Plot the evaluated point in 3D
    ax3d2.scatter([x_point[0]], [x_point[1]], [function_value], 
                 color='blue', s=100, marker='o')
    
    # Plot gradient vector in 3D
    ax3d2.quiver(x_point[0], x_point[1], function_value, 
                gradient_value[0], gradient_value[1], 0,
                color='red', arrow_length_ratio=0.3, linewidth=3)
    
    ax3d2.set_title('3D: Gradient at a Point')
    ax3d2.set_xlabel('x₁')
    ax3d2.set_ylabel('x₂')
    ax3d2.set_zlabel('f(x)')
    
    # 5. 2D Gradient Descent Path
    plt.subplot(3, 2, 5)
    plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=8)
    plt.title('Gradient Descent Path (2D)')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    
    # 6. 3D Gradient Descent Path
    ax3d3 = plt.subplot(3, 2, 6, projection='3d')
    ax3d3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # Plot the optimization path in 3D
    ax3d3.plot(trajectory[:, 0], trajectory[:, 1], function_values, 'ro-', linewidth=2, markersize=8)
    
    ax3d3.set_title('Gradient Descent Path (3D)')
    ax3d3.set_xlabel('x₁')
    ax3d3.set_ylabel('x₂')
    ax3d3.set_zlabel('f(x)')
    
    plt.tight_layout()
    plt.savefig('quadratic_analysis_complete.png', dpi=300)
    plt.show()
    
    # Analytical solution: The minimum of x^T A x is at x = 0
    minimum_point = np.array([0.0, 0.0])
    minimum_value = quadratic_function(minimum_point)
    print(f"\nAnalytical minimum: x = {minimum_point}, f(x) = {minimum_value}")

if __name__ == "__main__":
    main()

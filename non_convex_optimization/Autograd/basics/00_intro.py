import autograd.numpy as np  # Wraps NumPy's API to track operations for automatic differentiation
from autograd import grad, elementwise_grad  # Functions to compute derivatives
import matplotlib.pyplot as plt  # For visualization

# Autograd basics with visualization
def main():
    # Part 1: Define functions using autograd.numpy
    def f(x):
        """Simple scalar function: f(x) = x^2 + sin(x)"""
        return x**2 + np.sin(x)
    
    # Part 2: Computing gradients
    # grad takes a function and returns a function that computes the gradient
    f_grad = grad(f)  # df/dx
    
    # Part 3: Evaluate the function and its gradient at specific points
    x_value = 2.0
    print(f"f({x_value}) = {f(x_value)}")
    print(f"f'({x_value}) = {f_grad(x_value)}")
    
    # Part 4: Working with vectors and matrices
    def multivariate_function(x):
        """Function that takes a vector and returns a scalar"""
        return np.sum(x**2) + np.sin(x[0])
    
    multivariate_grad = grad(multivariate_function)
    
    x_vector = np.array([1.0, 2.0, 3.0])
    print(f"\nMultivariate function at {x_vector} = {multivariate_function(x_vector)}")
    print(f"Gradient at {x_vector} = {multivariate_grad(x_vector)}")
    
    # Part 5: Higher-order derivatives
    # Computing the second derivative (d²f/dx²)
    f_hessian = grad(f_grad)
    print(f"\nSecond derivative at {x_value} = {f_hessian(x_value)}")
    
    # Part 6: Visualization of function and its derivatives
    x_range = np.linspace(-3, 3, 1000)
    y_f = np.array([f(x_i) for x_i in x_range])
    y_grad = np.array([f_grad(x_i) for x_i in x_range])
    y_hessian = np.array([f_hessian(x_i) for x_i in x_range])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(x_range, y_f, 'b-', linewidth=2)
    plt.title('Function f(x) = x² + sin(x)')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(x_range, y_grad, 'r-', linewidth=2)
    plt.title('First Derivative f\'(x)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(x_range, y_hessian, 'g-', linewidth=2)
    plt.title('Second Derivative f\'\'(x)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('function_derivatives.png')
    plt.show()
    
    # Part 7: Gradient descent example with visualization
    def squared_error(x):
        """Function to minimize: f(x) = x^2"""
        return x**2
    
    squared_error_grad = grad(squared_error)
    
    # Implement gradient descent
    x = 5.0  # Starting point
    learning_rate = 0.1
    
    print("\nGradient descent example:")
    print(f"Starting x = {x}, f(x) = {squared_error(x)}")
    
    # For visualization
    x_points = [x]
    f_points = [squared_error(x)]
    
    for i in range(10):
        # Compute gradient at current x
        dx = squared_error_grad(x)
        
        # Update x using gradient descent
        x = x - learning_rate * dx
        
        # Store for visualization
        x_points.append(x)
        f_points.append(squared_error(x))
        
        # Print progress every few steps
        if i % 2 == 0:
            print(f"Step {i+1}: x = {x}, f(x) = {squared_error(x)}")
    
    # Visualize gradient descent
    x_vis = np.linspace(-5, 5, 100)
    y_vis = np.array([squared_error(x_i) for x_i in x_vis])
    
    plt.figure(figsize=(10, 6))
    
    # Plot the function
    plt.plot(x_vis, y_vis, 'b-', linewidth=2, label='f(x) = x²')
    
    # Plot the path of gradient descent
    plt.plot(x_points, f_points, 'ro-', linewidth=1.5, markersize=8, label='Gradient Descent Path')
    
    plt.title('Gradient Descent Optimization')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig('gradient_descent.png')
    plt.show()
    
    # Part 8: Visualize contour plot for multivariate function
    def simple_multivariate(xy):
        """A simple multivariate function f(x,y) = x² + y² + sin(x*y)"""
        x, y = xy
        return x**2 + y**2 + np.sin(x*y)
    
    # Create grid for visualization
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    # Compute function values
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            Z[j, i] = simple_multivariate([X[j, i], Y[j, i]])
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(label='f(x,y)')
    plt.contour(X, Y, Z, 20, colors='white', alpha=0.5, linestyles='solid', linewidths=0.5)
    
    plt.title('Contour Plot of f(x,y) = x² + y² + sin(x*y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig('contour_plot.png')
    plt.show()

if __name__ == "__main__":
    main()

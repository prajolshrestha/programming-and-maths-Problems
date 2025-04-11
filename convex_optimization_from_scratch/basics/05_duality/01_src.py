import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

"""
Simple Non-Convex Problem and Duality Demonstration

This script demonstrates:
1. A simple non-convex optimization problem
2. Visualization of the primal problem
3. Formulation of the dual problem
4. The duality gap in non-convex optimization
"""

# =============================================================================
# Part 1: Define a simple non-convex problem
# =============================================================================

def primal_problem():
    """
    A simple non-convex problem: Minimizing a non-convex function
    
    min f(x) = x²(1-x)²
    
    This has a non-convex objective function with multiple local minima.
    """
    
    # Create a figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define the non-convex function
    def f(x):
        return x**2 * (1-x)**2
    
    # Generate data for plotting
    x_vals = np.linspace(-0.5, 1.5, 1000)
    y_vals = f(x_vals)
    
    # Plot the function
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Non-Convex Function: f(x) = x²(1-x)²')
    ax1.grid(True)
    
    # Mark the global minimum at x=0.5
    ax1.plot(0.5, f(0.5), 'ro', markersize=8)
    ax1.text(0.5, f(0.5)+0.01, 'Global Minimum (0.5, 0.0625)', 
             horizontalalignment='center', color='red')
    
    # Highlight the non-convexity 
    p1 = 0.0
    p2 = 1.0
    mid_point = (p1 + p2) / 2
    
    ax1.plot([p1, p2], [f(p1), f(p2)], 'g--', linewidth=2, 
            label='Line between two points')
    ax1.plot([p1], [f(p1)], 'go', markersize=8)
    ax1.plot([p2], [f(p2)], 'go', markersize=8)
    ax1.plot([mid_point], [f(mid_point)], 'ro', markersize=8)
    
    # Add an arrow showing the function value at midpoint is below the line
    mid_y_on_line = (f(p1) + f(p2)) / 2
    ax1.annotate('', xy=(mid_point, f(mid_point)), 
                xytext=(mid_point, mid_y_on_line),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    ax1.text(mid_point, mid_y_on_line+0.02, 'f(midpoint) < midpoint of f', 
             horizontalalignment='center', color='red')
    
    ax1.legend(loc='upper right')
    
    # Now show a 3D version of a related 2D non-convex function
    def f2d(x, y):
        return (x**2 + y**2) * ((1-x)**2 + (1-y)**2)
    
    # Generate data for the 3D plot
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f2d(X, Y)
    
    # Create the 3D plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                           linewidth=0, antialiased=True)
    
    # Mark the global minimum
    ax2.scatter([0.5], [0.5], [f2d(0.5, 0.5)], color='red', s=100, marker='o')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('2D Non-Convex Function')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# =============================================================================
# Part 2: Lagrangian Duality Demonstration
# =============================================================================

def duality_demonstration():
    """
    Demonstrate duality for a constrained version of our non-convex problem.
    
    Primal Problem:
    min f(x) = x²(1-x)²
    s.t. x >= 0
         x <= 1
    
    This will show the Lagrangian function and the dual function.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define the non-convex function
    def f(x):
        return x**2 * (1-x)**2
    
    # Original function with constraints
    x_vals = np.linspace(-0.5, 1.5, 1000)
    y_vals = f(x_vals)
    
    # Plot the function
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x²(1-x)²')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Primal Problem with Constraints')
    ax1.grid(True)
    
    # Highlight the feasible region
    feasible_x = np.linspace(0, 1, 500)
    feasible_y = f(feasible_x)
    ax1.fill_between(feasible_x, 0, feasible_y, alpha=0.3, color='green',
                    label='Feasible Region')
    
    # Mark the global minimum
    ax1.plot(0.5, f(0.5), 'ro', markersize=8)
    ax1.text(0.5, f(0.5)+0.01, 'Global Minimum (0.5, 0.0)', 
             horizontalalignment='center', color='red')
    
    ax1.legend(loc='upper right')
    
    # Now demonstrate the Lagrangian and dual function
    # Lagrangian: L(x, λ₁, λ₂) = f(x) - λ₁x + λ₂(x-1)
    
    # For visualization, we'll fix λ₁ = 0 and vary λ₂
    def dual_function(lambda2):
        """Compute the dual function g(λ₂) with λ₁ = 0"""
        # For each λ₂, we need to minimize L(x, 0, λ₂) over x
        min_val = float('inf')
        min_x = None
        
        # Brute force search for the minimum
        for x in np.linspace(-2, 2, 1000):
            lagrangian = f(x) + lambda2 * (x - 1)
            if lagrangian < min_val:
                min_val = lagrangian
                min_x = x
        
        return min_val, min_x
    
    # Compute the dual function for different values of λ₂
    lambda2_vals = np.linspace(-0.05, 0.05, 100)
    dual_vals = []
    min_xs = []
    
    for lambda2 in lambda2_vals:
        dual_val, min_x = dual_function(lambda2)
        dual_vals.append(dual_val)
        min_xs.append(min_x)
    
    # Plot the dual function
    ax2.plot(lambda2_vals, dual_vals, 'r-', linewidth=2, 
            label='Dual Function g(λ₂)')
    ax2.set_xlabel('λ₂')
    ax2.set_ylabel('g(λ₂)')
    ax2.set_title('Dual Function (with λ₁=0)')
    ax2.grid(True)
    
    # Mark the maximum of the dual function
    max_idx = np.argmax(dual_vals)
    max_lambda2 = lambda2_vals[max_idx]
    max_dual = dual_vals[max_idx]
    
    ax2.plot(max_lambda2, max_dual, 'go', markersize=8)
    ax2.text(max_lambda2, max_dual+0.001, f'Max Dual Value: {max_dual:.4f}', 
             horizontalalignment='center', color='green')
    
    # Highlight the duality gap
    primal_optimal = f(0.5)  # The true minimum of the primal
    ax2.axhline(y=primal_optimal, color='b', linestyle='--', 
               label=f'Primal Optimal: {primal_optimal:.4f}')
    
    # Show the duality gap
    ax2.annotate('Duality Gap', xy=(max_lambda2, (max_dual + primal_optimal)/2),
                xytext=(max_lambda2 + 0.02, (max_dual + primal_optimal)/2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# =============================================================================
# Part 3: A more complicated example with visualizations in 2D
# =============================================================================

def constraint_visualization():
    """
    Visualize a non-convex constraint and its impact on duality
    
    Primal Problem:
    min f(x,y) = x² + y²
    s.t. (x-2)² + y² >= 1
    
    The constraint is non-convex (outside a circle)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define the domain
    x = np.linspace(-1, 3, 300)
    y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(x, y)
    
    # Objective function (convex): f(x,y) = x² + y²
    Z = X**2 + Y**2
    
    # Non-convex constraint: (x-2)² + y² >= 1
    constraint = (X-2)**2 + Y**2 >= 1
    
    # Create a mask for the feasible region
    feasible_mask = constraint
    
    # Plot contours of the objective function
    # These contours are level sets of the objective function Z = X**2 + Y**2
    # Each contour line represents points where the function has the same value
    contours = ax1.contour(X, Y, Z, levels=np.arange(0, 10, 1), colors='blue', alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=8)
    
    # Color the feasible region
    ax1.contourf(X, Y, feasible_mask, levels=[0, 0.5, 1], colors=['white', 'green'], alpha=0.3)
    
    # Draw the constraint boundary
    constraint_boundary = ax1.contour(X, Y, (X-2)**2 + Y**2, levels=[1], colors=['red'], 
                                    linewidths=2, linestyles='dashed')
    
    # Mark the optimal solution (analytically calculated)
    optimal_x = 1
    optimal_y = 0
    ax1.scatter(optimal_x, optimal_y, color='red', s=100, marker='*', 
              label=f'Optimal: ({optimal_x}, {optimal_y})')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Primal Problem with Non-Convex Constraint')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Add circle to illustrate the non-convex constraint
    circle = Circle((2, 0), 1, fill=False, edgecolor='red', linestyle='--', 
                  linewidth=2, label='Constraint Boundary')
    ax1.add_patch(circle)
    ax1.text(2, 0, 'Forbidden\nRegion', ha='center', color='red')
    
    # Now visualize the Lagrangian dual for this problem
    # Lagrangian: L(x,y,λ) = x² + y² - λ((x-2)² + y² - 1)
    
    # For various values of λ, find the minimum of the Lagrangian
    lambda_vals = np.linspace(0, 1, 100)
    dual_vals = []
    
    for lambda_val in lambda_vals:
        # For fixed λ, the minimum of the Lagrangian can be found analytically:
        # ∇L = 0 => (x,y) = (2λ/(1-λ), 0) for λ < 1
        # 
        # Steps to find this minimum:
        # 1. Lagrangian: L(x,y,λ) = x² + y² - λ((x-2)² + y² - 1)
        # 2. Expand: L(x,y,λ) = x² + y² - λ(x² - 4x + 4 + y² - 1)
        # 3. Simplify: L(x,y,λ) = (1-λ)x² + (1-λ)y² + 4λx - 3λ
        # 4. Partial derivatives: ∂L/∂x = 2(1-λ)x + 4λ = 0, ∂L/∂y = 2(1-λ)y = 0
        # 5. From ∂L/∂y = 0: either y = 0 or λ = 1
        # 6. From ∂L/∂x = 0: 2(1-λ)x + 4λ = 0
        # 7. Solving for x: 2(1-λ)x = -4λ, so x = -4λ/[2(1-λ)] = -2λ/(1-λ) when λ < 1
        # 8. The negative sign is correct and comes directly from solving the equation
        if lambda_val < 1:
            min_x = -2 * lambda_val / (1 - lambda_val)
            min_y = 0
            dual_val = min_x**2 + min_y**2 - lambda_val * ((min_x-2)**2 + min_y**2 - 1)
        else:
            # When λ ≥ 1, the Lagrangian becomes unbounded below
            # This is because the coefficient of x² and y² becomes (1-λ) which is negative when λ > 1
            # Making the quadratic terms negative means we can make L(x,y,λ) arbitrarily small
            # by choosing sufficiently large values of x or y
            dual_val = -np.inf
        dual_vals.append(dual_val)
    
    # The calculation of the dual function g(λ) = inf L(x,y,λ) 
    # based on substituting these x and y values back into L, 
    # resulting in g(λ) = (-3λ - λ²)/(1-λ).
    
    # Plot the dual function
    valid_mask = np.isfinite(dual_vals)
    ax2.plot(lambda_vals[valid_mask], np.array(dual_vals)[valid_mask], 'r-', linewidth=2, 
            label='Dual Function g(λ)')
    
    # Mark the maximum of the dual function
    if any(valid_mask):
        max_idx = np.argmax(np.array(dual_vals)[valid_mask])
        max_lambda = lambda_vals[valid_mask][max_idx]
        max_dual = np.array(dual_vals)[valid_mask][max_idx]
        
        ax2.plot(max_lambda, max_dual, 'go', markersize=8)
        ax2.text(max_lambda, max_dual+0.05, f'Max Dual: {max_dual:.4f}', 
                 ha='center', color='green')
    
    # Show the primal optimal value
    primal_optimal = optimal_x**2 + optimal_y**2  # = 1
    ax2.axhline(y=primal_optimal, color='b', linestyle='--', 
               label=f'Primal Optimal: {primal_optimal:.4f}')
    
    # Show the duality gap if it exists
    if any(valid_mask) and max_dual < primal_optimal:
        gap = primal_optimal - max_dual
        ax2.annotate(f'Duality Gap: {gap:.4f}', 
                    xy=(max_lambda, (max_dual + primal_optimal)/2),
                    xytext=(max_lambda + 0.2, (max_dual + primal_optimal)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax2.set_xlabel('λ')
    ax2.set_ylabel('g(λ)')
    ax2.set_title('Dual Function and Duality Gap')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Run all demonstrations
    print("Demonstrating a simple non-convex optimization problem...")
    primal_problem()
    
    print("\nDemonstrating Lagrangian duality for a constrained non-convex problem...")
    duality_demonstration()
    
    print("\nVisualizing non-convex constraints and duality gap...")
    constraint_visualization()
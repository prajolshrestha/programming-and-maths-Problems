"""
The Color Mixing Story: Understanding Least Squares Optimization
With 3D Visualization
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def color_mixing_example(with_constraints=False):
    # Setup our color space (3D for RGB visualization)
    np.random.seed(42)
    num_base_colors = 3
    color_space_dim = 3
    
    # Generate random base colors (A) and target color (b)
    A = np.random.rand(color_space_dim, num_base_colors)  # Each column is a base color
    b = np.random.rand(color_space_dim)  # Target color
    
    # Define our mixing ratios variable
    x = cp.Variable(num_base_colors)
    
    # Define the objective: minimize distance to target color
    cost = cp.sum_squares(A @ x - b)
    
    if with_constraints:
        constraints = [
            x >= 0,              # Non-negative amounts of paint
            cp.sum(x) <= 1,      # Total amount of paint ≤ 1
            x <= 0.5             # No more than 0.5 units of any color
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)
    else:
        prob = cp.Problem(cp.Minimize(cost))
    
    prob.solve()
    
    return {
        'base_colors': A,
        'target_color': b,
        'mixing_ratios': x.value,
        'achieved_color': A @ x.value,
        'color_difference': np.linalg.norm(A @ x.value - b)
    }

def visualize_color_mixing(unconstrained, constrained):
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Base colors and results (unconstrained)
    ax1 = fig.add_subplot(131, projection='3d')
    plot_color_space(ax1, unconstrained, "Unconstrained Mixing")
    
    # Plot 2: Base colors and results (constrained)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_color_space(ax2, constrained, "Constrained Mixing")
    
    # Plot 3: Comparison of mixing ratios
    ax3 = fig.add_subplot(133)
    plot_mixing_ratios(ax3, unconstrained, constrained)
    
    plt.tight_layout()
    plt.show()

def plot_color_space(ax, result, title):
    # Plot base colors
    base_colors = result['base_colors']
    for i in range(base_colors.shape[1]):
        ax.scatter([base_colors[0,i]], [base_colors[1,i]], [base_colors[2,i]], 
                  c='rgb'[i], s=100, label=f'Base Color {i+1}')
    
    # Plot target color
    target = result['target_color']
    ax.scatter([target[0]], [target[1]], [target[2]], 
              c='black', s=100, label='Target Color')
    
    # Plot achieved color
    achieved = result['achieved_color']
    ax.scatter([achieved[0]], [achieved[1]], [achieved[2]], 
              c='purple', s=100, label='Achieved Color')
    
    # Draw line from achieved to target color (showing error)
    ax.plot([achieved[0], target[0]], 
            [achieved[1], target[1]], 
            [achieved[2], target[2]], 
            'r--', label='Error')
    
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title(title)
    ax.legend()

def plot_mixing_ratios(ax, unconstrained, constrained):
    x = range(len(unconstrained['mixing_ratios']))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], unconstrained['mixing_ratios'], 
           width, label='Unconstrained', color='blue', alpha=0.6)
    ax.bar([i + width/2 for i in x], constrained['mixing_ratios'], 
           width, label='Constrained', color='red', alpha=0.6)
    
    ax.set_ylabel('Mixing Ratio')
    ax.set_title('Mixing Ratios Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Red', 'Green', 'Blue'])
    ax.legend()

# Run both versions and visualize
unconstrained = color_mixing_example(with_constraints=False)
constrained = color_mixing_example(with_constraints=True)

# Print numerical results
print("Unconstrained Mixing:")
print(f"Mixing ratios: {unconstrained['mixing_ratios']}")
print(f"Color difference: {unconstrained['color_difference']:.4f}")
print("\nConstrained Mixing:")
print(f"Mixing ratios: {constrained['mixing_ratios']}")
print(f"Color difference: {constrained['color_difference']:.4f}")

# Create visualizations
visualize_color_mixing(unconstrained, constrained)

"""
Visualization Explanation:

1. 3D Color Space Plots (Left and Middle):
   - Black dot: Target color (b)
   - RGB dots: Base colors (columns of A)
   - Purple dot: Achieved color (Ax)
   - Red dashed line: Error between target and achieved color
   
2. Mixing Ratios Bar Plot (Right):
   - Shows how much of each base color is used
   - Compares constrained vs unconstrained solutions
   - Negative values only appear in unconstrained version
   
Key Observations:
1. Unconstrained mixing usually gets closer to the target
2. Constrained mixing keeps ratios positive and bounded
3. The error (dashed line) shows how close we got to the target
""" 
"""
Second Order Cone (SOC) Intuition:

1. A second-order cone in 2D forms a "ice cream cone" shape where:
   - The vertical axis represents t
   - The horizontal plane represents (x,y)
   - Points inside satisfy: t ≥ √(x² + y²)

2. Circle Constraint Connection:
   - When we constrain ||new_pos - old_pos||₂ ≤ d_max
   - This forms a circle in 2D with radius d_max
   - The circle is actually a slice of the 3D cone at t = d_max
   - All points in the circle satisfy: d_max ≥ √(x² + y²)

3. Geometric Interpretation:
   - The cone extends upward in the t dimension
   - Any horizontal slice of the cone is a circle
   - The radius of the circle equals the height of the slice
   - The circle constraint is just one such slice

4. Example Application:
   - Used in robot motion planning to limit maximum step size
   - Helps maintain smooth trajectories
   - Ensures physically feasible movements
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_soc_distance_constraint(old_pos, d_max, num_points=1000):
    """
    Visualize the 2D SOC constraint: ||new_pos - old_pos||₂ ≤ d_max
    
    Parameters:
    -----------
    old_pos: array-like
        Current position [x₀, y₀]
    d_max: float
        Maximum allowed movement distance
    num_points: int
        Number of random points to generate for visualization
    """
    old_pos = np.array(old_pos)
    
    # Generate random points within a unit circle using rejection sampling
    points = []
    while len(points) < num_points:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            points.append((x, y))
    
    points = np.array(points)
    
    # Scale points by d_max and shift to old_pos
    points = points * d_max + old_pos
    
    # Create the figure
    plt.figure(figsize=(8, 8))
    
    # Plot the feasible region (interior points)
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.3, s=5, label='Feasible Region')
    
    # Plot the current position
    plt.scatter([old_pos[0]], [old_pos[1]], color='red', s=200, marker='*', label='Current Position')
    
    # Plot the boundary circle
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = d_max * np.cos(theta) + old_pos[0]
    y_circle = d_max * np.sin(theta) + old_pos[1]
    plt.plot(x_circle, y_circle, 'b-', linewidth=2, label='Maximum Distance')
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'2D SOC Constraint: New Position Within Distance {d_max} of Current Position')
    plt.grid(True)
    plt.axis('equal')
    
    # Add legend
    plt.legend()
    
    return plt

def visualize_circle_cone_relationship(d_max=2.0):
    """
    Visualize how a 2D circular constraint relates to a 3D second-order cone.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate the cone
    t_max = d_max * 1.5  # Go a bit beyond our constraint
    resolution = 50
    
    # Create a meshgrid for the circular cross-sections
    theta = np.linspace(0, 2*np.pi, resolution)
    t_values = np.linspace(0, t_max, resolution)
    
    # Create points for the cone surface
    cone_x = []
    cone_y = []
    cone_z = []
    
    for t in t_values:
        radius = t  # In a standard SOC, radius equals height
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        cone_x.extend(x)
        cone_y.extend(y)
        cone_z.extend([t] * len(x))
    
    # Plot the cone as a surface
    ax.scatter(cone_x, cone_y, cone_z, c='blue', alpha=0.1, s=5)
    
    # Plot the constraint plane at t = d_max
    xx, yy = np.meshgrid(np.linspace(-d_max, d_max, 10), 
                         np.linspace(-d_max, d_max, 10))
    ax.plot_surface(xx, yy, d_max * np.ones_like(xx), 
                   alpha=0.3, color='red')
    
    # Plot the resulting circle at the intersection
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = d_max * np.cos(circle_theta)
    circle_y = d_max * np.sin(circle_theta)
    circle_z = d_max * np.ones_like(circle_theta)
    ax.plot(circle_x, circle_y, circle_z, 'g-', linewidth=4)
    
    # Add labels and explanations
    ax.set_xlabel('x - x₀')
    ax.set_ylabel('y - y₀')
    ax.set_zlabel('t (auxiliary variable)')
    ax.set_title('Relationship Between 2D Circle Constraint and 3D Second-Order Cone')
    
    # Add text explanations
    ax.text(0, 0, 0, "Origin", color='black', fontsize=12)
    ax.text(0, 0, d_max, f"t = d_max = {d_max}", color='red', fontsize=12)
    ax.text(d_max, 0, d_max, "Circle constraint boundary", color='green', fontsize=12)
    
    # Highlight the origin and the axis
    ax.scatter([0], [0], [0], color='black', s=100)
    z_axis = np.linspace(0, t_max, 10)
    ax.plot([0]*10, [0]*10, z_axis, 'k--', linewidth=2)
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    return plt

# Example usage
old_pos = [2, 3]   # Current position
d_max = 1.5        # Maximum allowed movement distance

plt = visualize_2d_soc_distance_constraint(old_pos, d_max)
#plt.savefig('2d_soc_distance_constraint.png', dpi=300)
plt.show()

# Visualize the relationship
plt = visualize_circle_cone_relationship(d_max=2.0)
#plt.savefig('circle_cone_relationship.png', dpi=300)
plt.show()

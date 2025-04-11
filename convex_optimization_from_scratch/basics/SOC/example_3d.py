"""
Second Order Cone (SOC) Intuition in Higher Dimensions:

1. 3D Second-Order Cone:
   - Forms a cone shape in 4D space (t, x, y, z)
   - Points inside satisfy: t ≥ √(x² + y² + z²)
   - The "ice cream cone" visualization extends to 4D
   - Cross sections perpendicular to t-axis are 3D spheres

2. Sphere Constraint Connection:
   - When we constrain ||new_pos - old_pos||₂ ≤ d_max in 3D
   - This forms a sphere with radius d_max
   - All feasible new positions lie within this sphere
   - The sphere is a slice of the 4D cone at t = d_max

3. Geometric Interpretation:
   - The cone extends upward in the t dimension
   - Any horizontal slice of the cone is a sphere
   - The radius of the sphere equals the height of the slice
   - The sphere constraint is just one such slice
   - As t increases, the sphere radius grows linearly
   - The cone's surface has constant slope in all directions
   - Points inside the cone represent feasible solutions
   - The cone's apex is at the origin (0,0,0,0)

4. Example Applications:
   - Robot Motion Planning: Limit 3D movement steps
   - Signal Processing: Bound signal power/energy
   - Portfolio Optimization: Risk-return trade-offs
   - Antenna Design: Radiation pattern constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_soc_distance_constraint(old_pos, d_max, num_points=1000):
    """
    Visualize the SOC constraint: ||new_pos - old_pos||₂ ≤ d_max
    
    Parameters:
    -----------
    old_pos: array-like
        Current position [x₀, y₀, z₀]
    d_max: float
        Maximum allowed movement distance
    num_points: int
        Number of random points to generate for visualization
    """
    old_pos = np.array(old_pos)
    
    # Generate random points on a unit sphere
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    # Convert to Cartesian coordinates on unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Scale points by random factors to fill the sphere
    r = np.random.uniform(0, 1, num_points)**(1/3)  # Cube root for uniform distribution in a sphere
    x = x * r * d_max
    y = y * r * d_max
    z = z * r * d_max
    
    # Shift points to be centered at old_pos
    x = x + old_pos[0]
    y = y + old_pos[1]
    z = z + old_pos[2]
    
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points within the sphere
    ax.scatter(x, y, z, c='blue', alpha=0.3, s=5)
    
    # Plot the old position (center of sphere)
    ax.scatter([old_pos[0]], [old_pos[1]], [old_pos[2]], color='red', s=200, label='Current Position')
    
    # Add wireframe to visualize the sphere boundary
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = d_max * np.outer(np.cos(u), np.sin(v)) + old_pos[0]
    y_sphere = d_max * np.outer(np.sin(u), np.sin(v)) + old_pos[1]
    z_sphere = d_max * np.outer(np.ones(np.size(u)), np.cos(v)) + old_pos[2]
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.1)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'SOC Constraint: New Position Within Distance {d_max} of Current Position')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    return plt

# Example usage
old_pos = [1, 2, 3]  # Current position
d_max = 2.5         # Maximum allowed movement distance

plt = visualize_soc_distance_constraint(old_pos, d_max)
#plt.savefig('soc_distance_constraint.png', dpi=300)
plt.show()

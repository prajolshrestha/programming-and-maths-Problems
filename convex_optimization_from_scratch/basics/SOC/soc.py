import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_cone(angle_degrees=45, height=5, num_points=100):
    """
    Visualize a 2D cone with the specified angle.
    
    Parameters:
    -----------
    angle_degrees : float
        Half-angle of the cone in degrees
    height : float
        Height of the cone
    num_points : int
        Number of points for visualization
    """
    angle_rad = np.radians(angle_degrees)
    
    # Calculate the radius at the top
    radius = height * np.tan(angle_rad)
    
    # Create the cone boundary lines
    x_left = np.linspace(0, radius, num_points)
    y_left = np.linspace(0, height, num_points)
    
    x_right = np.linspace(0, -radius, num_points)
    y_right = np.linspace(0, height, num_points)
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    plt.plot(x_left, y_left, 'b-', linewidth=2)
    plt.plot(x_right, y_right, 'b-', linewidth=2)
    
    # Fill the cone
    x_fill = np.concatenate([x_right, x_left[::-1]])
    y_fill = np.concatenate([y_right, y_left[::-1]])
    plt.fill(x_fill, y_fill, alpha=0.3, color='blue')
    
    # Add the origin point
    plt.plot(0, 0, 'ro')
    
    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'2D Cone with {angle_degrees}° Half-Angle')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-height, height)
    plt.ylim(-1, height + 1)
    plt.tight_layout()
    
    return plt

def visualize_3d_cone(angle_degrees=45, height=5, resolution=50):
    """
    Visualize a 3D cone with the specified angle.
    
    Parameters:
    -----------
    angle_degrees : float
        Half-angle of the cone in degrees
    height : float
        Height of the cone
    resolution : int
        Resolution of the cone mesh
    """
    angle_rad = np.radians(angle_degrees)
    
    # Calculate the radius at the top
    radius = height * np.tan(angle_rad)
    
    # Create a meshgrid for the base of the cone
    theta = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, resolution)
    T, R = np.meshgrid(theta, r)
    
    # Convert to Cartesian coordinates
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = height * R / radius  # Linear scaling to keep the cone shape
    
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the cone surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Plot the wireframe
    ax.plot_wireframe(X, Y, Z, color='k', alpha=0.2, rcount=10, ccount=10)
    
    # Plot the origin
    ax.scatter([0], [0], [0], color='red', s=100)
    
    # Add some points along the principal axis
    z_axis = np.linspace(0, height, 10)
    ax.plot([0]*10, [0]*10, z_axis, 'r--', linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Cone with {angle_degrees}° Half-Angle')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return plt

def visualize_second_order_cone(n=3, num_points=1000):
    """
    Visualize a Second Order Cone in n dimensions: {(t, x) ∈ ℝ × ℝⁿ⁻¹ | t ≥ ||x||₂}
    For n=2, this is a 2D SOC (ice cream cone)
    For n=3, this is a 3D SOC (Lorentz cone)
    
    Parameters:
    -----------
    n : int
        Dimension of the cone (2 or 3 for visualization)
    num_points : int
        Number of points for sampling
    """
    if n == 2:
        # 2D Second Order Cone (ice cream cone)
        t = np.linspace(0, 5, 100)
        
        # Left boundary: t = -x
        x_left = -t
        # Right boundary: t = x
        x_right = t
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_left, t, 'b-', linewidth=2)
        plt.plot(x_right, t, 'b-', linewidth=2)
        
        # Fill the cone
        x_fill = np.concatenate([x_left, x_right[::-1]])
        t_fill = np.concatenate([t, t[::-1]])
        plt.fill(x_fill, t_fill, alpha=0.3, color='blue')
        
        # Add the origin point
        plt.plot(0, 0, 'ro')
        
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('2D Second Order Cone: {(t, x) ∈ ℝ² | t ≥ |x|}')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-1, 6)
        
    elif n == 3:
        # 3D Second Order Cone (Lorentz cone)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate random points on a unit circle
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = np.random.uniform(0, 1, num_points)**(1/2)  # Square root for uniform distribution
        
        # Convert to Cartesian coordinates in the x-y plane
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale up to fill the cone (t ≥ √(x² + y²))
        max_t = 5
        # Scale factors for each point
        scales = np.random.uniform(0, 1, num_points)
        
        # Final coordinates with scaling
        x_scaled = x * scales * max_t
        y_scaled = y * scales * max_t
        t_scaled = np.sqrt(x_scaled**2 + y_scaled**2 + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Plot the points
        ax.scatter(x_scaled, y_scaled, t_scaled, c=t_scaled, cmap='viridis', alpha=0.8, s=5)
        
        # Add wireframe to visualize the cone boundary
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, max_t, 10)
        u, v = np.meshgrid(u, v)
        
        x_wire = v * np.cos(u) * 0.99  # Scale slightly to avoid overlap
        y_wire = v * np.sin(u) * 0.99
        z_wire = np.sqrt(x_wire**2 + y_wire**2)
        
        ax.plot_wireframe(x_wire, y_wire, z_wire, color='k', alpha=0.2)
        
        # Plot the origin
        ax.scatter([0], [0], [0], color='red', s=100)
        
        # Plot the z-axis
        z_axis = np.linspace(0, max_t, 10)
        ax.plot([0]*10, [0]*10, z_axis, 'r--', linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('t')
        ax.set_title('3D Second Order Cone: {(t, x, y) ∈ ℝ³ | t ≥ √(x² + y²)}')
        ax.set_box_aspect([1, 1, 1])
        
    else:
        raise ValueError("Visualization only supports 2D and 3D cones (n=2 or n=3)")
    
    return plt

if __name__ == "__main__":
    # Visualize geometric cones
    plt1 = visualize_2d_cone(angle_degrees=30, height=5)
    #plt1.savefig('2d_geometric_cone.png', dpi=300)
    plt1.show()
    
    plt2 = visualize_3d_cone(angle_degrees=30, height=5)
    #plt2.savefig('3d_geometric_cone.png', dpi=300)
    plt2.show()
    
    # Visualize second-order cones used in convex optimization
    plt3 = visualize_second_order_cone(n=2)
   # plt3.savefig('2d_second_order_cone.png', dpi=300)
    plt3.show()
    
    plt4 = visualize_second_order_cone(n=3)
    #plt4.savefig('3d_second_order_cone.png', dpi=300)
    plt4.show()
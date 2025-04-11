import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_color_cone(base_colors, num_points=1000):
    """
    Create points in a convex cone using base colors.
    
    Args:
        base_colors: List of RGB colors as base vectors
        num_points: Number of random points to generate
    
    Returns:
        Array of points in the convex cone
    """
    # Convert base colors to numpy array
    base_colors = np.array(base_colors)
    
    # Generate random weights (non-negative)
    weights = np.random.rand(num_points, len(base_colors))
    
    # Normalize weights to create varying intensities
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    
    # Generate points in the convex cone
    points = np.dot(weights, base_colors)
    return points

def plot_color_cone(points, base_colors):
    """
    Plot the color cone in 3D RGB space.
    
    Args:
        points: Array of points in the convex cone
        base_colors: List of base RGB colors
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the random points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=points, s=20)
    
    # Plot the base colors
    base_colors = np.array(base_colors)
    ax.scatter(base_colors[:, 0], base_colors[:, 1], base_colors[:, 2],
              c=base_colors, s=100, marker='*', label='Base Colors')
    
    # Plot lines from origin to base colors
    origin = np.zeros(3)
    for color in base_colors:
        ax.plot([origin[0], color[0]], 
                [origin[1], color[1]], 
                [origin[2], color[2]], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('Color Mixing as a Convex Cone in RGB Space')
    
    # Add legend
    ax.legend()
    
    plt.show()

def create_square_with_holes():
    """
    Create a visualization of a square with circular holes to demonstrate non-convex regions.
    Shows how moving between two points might require avoiding holes (non-convex path).
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Show the square with holes
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    
    # Draw the square
    square = plt.Rectangle((0, 0), 10, 10, fill=False, color='black')
    ax1.add_patch(square)
    
    # Draw three circular holes
    circle1 = plt.Circle((3, 3), 1.5, color='red', alpha=0.3)
    circle2 = plt.Circle((7, 7), 1.5, color='red', alpha=0.3)
    circle3 = plt.Circle((3, 7), 1.5, color='red', alpha=0.3)
    
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.add_patch(circle3)
    
    # Draw two points and a line between them
    point1 = [2, 2]
    point2 = [8, 8]
    
    ax1.plot(point1[0], point1[1], 'go', label='Start Point')
    ax1.plot(point2[0], point2[1], 'bo', label='End Point')
    
    # Draw straight line (which passes through holes)
    ax1.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--', 
             label='Direct Path (Invalid)')
    
    # Draw a valid curved path avoiding holes
    t = np.linspace(0, 1, 100)
    x = point1[0] + (point2[0] - point1[0]) * t
    y = point1[1] + (point2[1] - point1[1]) * t + 2 * np.sin(np.pi * t)
    ax1.plot(x, y, 'g-', label='Valid Path')
    
    ax1.set_title('Non-convex Region: Square with Holes')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Demonstrate convex combination
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-1, 11)
    
    # Draw the square
    square2 = plt.Rectangle((0, 0), 10, 10, fill=False, color='black')
    ax2.add_patch(square2)
    
    # Draw points
    point_a = [2, 2]
    point_b = [8, 8]
    ax2.plot(point_a[0], point_a[1], 'go', label='Point A')
    ax2.plot(point_b[0], point_b[1], 'bo', label='Point B')
    
    # Draw convex combinations
    t = np.linspace(0, 1, 10)
    for ti in t:
        x = point_a[0] * (1-ti) + point_b[0] * ti
        y = point_a[1] * (1-ti) + point_b[1] * ti
        ax2.plot(x, y, 'ko', alpha=0.3)
    
    # Draw line connecting points
    ax2.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 'k--', 
             label='Convex Combination')
    
    ax2.set_title('Convex Combination of Two Points')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Define base colors in RGB space
    base_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
    ]
    
    # Generate points in the convex cone
    points = create_color_cone(base_colors, num_points=1000)
    
    # Plot the color cone
    plot_color_cone(points, base_colors)
    
    create_square_with_holes()

if __name__ == "__main__":
    main()

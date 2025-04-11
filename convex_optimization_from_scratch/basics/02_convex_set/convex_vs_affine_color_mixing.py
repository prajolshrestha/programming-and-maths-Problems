"""
Understanding Convex vs Affine Sets through Color Mixing in 3D RGB Space

Convex Combination:
- Coefficients must sum to 1
- All coefficients must be non-negative
- Example: "70% red + 30% blue" (all positive)

Affine Combination:
- Coefficients must sum to 1
- Coefficients can be negative
- Example: "150% red - 50% blue" (can have negative)

Key Difference:
- Convex: All coefficients must be ≥ 0 (sum to 1)
- Affine: Coefficients can be negative (still sum to 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ColorMixer:
    def __init__(self):
        # Define base colors in RGB format
        self.red = np.array([1, 0, 0])      # Red
        self.blue = np.array([0, 0, 1])     # Blue
        self.yellow = np.array([1, 1, 0])   # Yellow
        
        self.base_colors = np.vstack([self.red, self.blue, self.yellow])
    
    def create_convex_combination(self, weights):
        """
        Create a color using convex combination
        weights must be non-negative and sum to 1
        """
        weights = np.array(weights)
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must be non-negative and sum to 1")
        return weights @ self.base_colors
    
    def create_affine_combination(self, weights):
        """
        Create a color using affine combination
        weights must sum to 1 (can be negative)
        """
        weights = np.array(weights)
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
        return weights @ self.base_colors

    def visualize_comparison(self, num_points=1000):
        """Compare convex and affine combinations"""
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Convex Set
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_convex_set(ax1, num_points)
        
        # 2. Affine Set
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_affine_set(ax2, num_points)
        
        # 3. Comparison
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_comparison(ax3, num_points)
        
        plt.tight_layout()
        plt.show()

    def _plot_convex_set(self, ax, num_points):
        """Plot convex combinations in RGB space"""
        # Plot base colors
        ax.scatter(self.base_colors[:, 0], 
                  self.base_colors[:, 1], 
                  self.base_colors[:, 2], 
                  c=['red', 'blue', 'yellow'], 
                  s=100, label='Base Colors')
        
        # Generate random convex combinations (non-negative, sum to 1)
        weights = np.random.rand(num_points, 3)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]
        
        combinations = weights @ self.base_colors
        
        # Plot combinations
        scatter = ax.scatter(combinations[:, 0], 
                           combinations[:, 1], 
                           combinations[:, 2],
                           c=combinations, 
                           alpha=0.6)
        
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title('Convex Set\n(Non-negative coefficients)')
        ax.legend()

    def _plot_affine_set(self, ax, num_points):
        """Plot affine combinations in RGB space"""
        # Plot base colors
        ax.scatter(self.base_colors[:, 0], 
                  self.base_colors[:, 1], 
                  self.base_colors[:, 2], 
                  c=['red', 'blue', 'yellow'], 
                  s=100, label='Base Colors')
        
        # Generate random affine combinations (can be negative, sum to 1)
        weights = np.random.randn(num_points, 3)  # Normal distribution for negative values
        weights = weights / weights.sum(axis=1)[:, np.newaxis]
        
        combinations = weights @ self.base_colors
        
        # Plot combinations
        ax.scatter(combinations[:, 0], 
                  combinations[:, 1], 
                  combinations[:, 2],
                  c='blue',
                  alpha=0.6,
                  label='Affine Combinations')
        
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title('Affine Set\n(Can have negative coefficients)')
        ax.legend()

    def _plot_comparison(self, ax, num_points):
        """Plot both sets for comparison"""
        # Generate convex combinations
        weights_convex = np.random.rand(num_points, 3)
        scaling = np.random.rand(num_points, 1)
        weights_convex = weights_convex / weights_convex.sum(axis=1)[:, np.newaxis] * scaling
        combinations_convex = weights_convex @ self.base_colors

        # Generate affine combinations
        weights_affine = np.random.rand(num_points, 3)
        weights_affine = weights_affine / weights_affine.sum(axis=1)[:, np.newaxis]
        combinations_affine = weights_affine @ self.base_colors

        # Plot both
        ax.scatter(combinations_convex[:, 0], 
                  combinations_convex[:, 1], 
                  combinations_convex[:, 2],
                  c=combinations_convex,
                  alpha=0.3,
                  label='Convex')
        
        ax.scatter(combinations_affine[:, 0], 
                  combinations_affine[:, 1], 
                  combinations_affine[:, 2],
                  c=combinations_affine,
                  alpha=0.3,
                  label='Affine')

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title('Comparison\nConvex vs Affine')
        ax.legend()

# Example usage
mixer = ColorMixer()

print("Example Color Combinations:")

# Convex combination (all positive)
print("\nConvex combination (all positive):")
print(mixer.create_convex_combination([0.7, 0.2, 0.1]))

# Affine combination (with negative)
print("\nAffine combination (with negative):")
print(mixer.create_affine_combination([1.5, -0.3, -0.2]))

try:
    # This will raise an error - negative weights not allowed in convex
    print("\nTrying invalid convex combination:")
    mixer.create_convex_combination([1.2, -0.1, -0.1])
except ValueError as e:
    print(f"Error: {e}")

# Visualize the differences
mixer.visualize_comparison()

"""
Key Insights:

1. Convex Combinations:
   - All coefficients must be non-negative
   - Coefficients must sum to 1
   - Forms a triangle (convex hull of base points)
   - More intuitive for physical mixing

2. Affine Combinations:
   - Coefficients can be negative
   - Coefficients must sum to 1
   - Extends beyond the triangle
   - More abstract mathematical concept

3. Main Difference:
   - Convex: Restricted to non-negative coefficients
   - Affine: Allows negative coefficients
   - Both require coefficients to sum to 1

4. Physical Interpretation:
   - Convex: Like real mixing (can't use negative paint)
   - Affine: Mathematical extension (can "subtract" colors)
"""
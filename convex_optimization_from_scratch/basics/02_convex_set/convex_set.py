"""
Understanding Convex Sets: The Recipe Analogy

A convex set is a set where any line segment between two points in the set lies entirely within the set.
Convex combination: Linear combination where coefficients are non-negative and sum to 1.

Recipe Analogy:
- Different recipes (points in the set) for making a dish
- New valid recipes must use non-negative proportions that sum to 100%
- Unlike affine sets, we can't use negative proportions (which makes more practical sense!)

Example: Imagine three basic smoothie recipes (points in our convex set)
- Recipe A: Berry Blast    (high berry, low yogurt)
- Recipe B: Creamy Classic (low berry, high yogurt)
- Recipe C: Balanced Blend (medium berry, medium yogurt)
"""

import numpy as np
import matplotlib.pyplot as plt

class SmoothieRecipe:
    def __init__(self):
        # Define three base recipes in terms of (berry, yogurt) proportions
        self.recipe_A = np.array([0.8, 0.2])  # Berry Blast: 80% berry, 20% yogurt
        self.recipe_B = np.array([0.2, 0.8])  # Creamy: 20% berry, 80% yogurt
        self.recipe_C = np.array([0.5, 0.5])  # Balanced: 50% berry, 50% yogurt
        
        self.base_recipes = np.vstack([self.recipe_A, self.recipe_B, self.recipe_C])
        
    def create_convex_combination(self, coefficients):
        """
        Create a new recipe using convex combination
        coefficients must be non-negative and sum to 1
        """
        if not np.isclose(sum(coefficients), 1.0):
            raise ValueError("Coefficients must sum to 1 (100%)")
        if not all(c >= 0 for c in coefficients):
            raise ValueError("All coefficients must be non-negative")
            
        return coefficients @ self.base_recipes
    
    def visualize_convex_set(self, num_points=1000):
        """
        Visualize the convex set by generating random convex combinations
        """
        # Create the original 2D plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot base recipes
        ax1.scatter(self.base_recipes[:, 0], self.base_recipes[:, 1], 
                   c='red', s=100, label='Base Recipes')
        ax1.annotate('Berry Blast (A)', (self.recipe_A[0], self.recipe_A[1]))
        ax1.annotate('Creamy (B)', (self.recipe_B[0], self.recipe_B[1]))
        ax1.annotate('Balanced (C)', (self.recipe_C[0], self.recipe_C[1]))
        
        # Generate random convex combinations
        random_recipes = []
        for _ in range(num_points):
            # Generate random non-negative coefficients that sum to 1
            coef = np.random.rand(3)
            coef = coef / coef.sum()
            
            new_recipe = self.create_convex_combination(coef)
            random_recipes.append(new_recipe)
            
        random_recipes = np.array(random_recipes)
        
        # Plot random convex combinations
        ax1.scatter(random_recipes[:, 0], random_recipes[:, 1], 
                   c='blue', alpha=0.1, label='Possible Recipes')
        
        ax1.set_xlabel('Berry Proportion')
        ax1.set_ylabel('Yogurt Proportion')
        ax1.set_title('Convex Set of Smoothie Recipes')
        
        # Demonstrate non-convex point
        non_convex_point = np.array([0.9, 0.9])  # Sum > 1, not possible!
        ax1.scatter(non_convex_point[0], non_convex_point[1], 
                   c='red', marker='x', s=100, label='Invalid Recipe')
        ax1.legend()
        ax1.grid(True)
        
        # Plot the coefficients triangle
        self._plot_coefficient_space(ax2)
        
        plt.tight_layout()
        
        # Create a new figure for 3D visualization
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        # Plot base recipes in 3D
        vertices_3d = np.array([
            [1, 0, 0],  # 100% Recipe A
            [0, 1, 0],  # 100% Recipe B
            [0, 0, 1]   # 100% Recipe C
        ])
        
        # Plot vertices
        ax3d.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], 
                    c='red', s=100, label='Base Recipes')
        
        # Generate points on the triangular plane
        random_points = []
        for _ in range(num_points):
            # Generate random non-negative coefficients that sum to 1
            coef = np.random.rand(3)
            coef = coef / coef.sum()
            random_points.append(coef)
        
        random_points = np.array(random_points)
        
        # Plot random points
        ax3d.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], 
                    c='blue', alpha=0.1, label='Recipe Combinations')
        
        # Plot the edges of the triangle
        for i in range(3):
            j = (i + 1) % 3
            ax3d.plot([vertices_3d[i,0], vertices_3d[j,0]],
                      [vertices_3d[i,1], vertices_3d[j,1]],
                      [vertices_3d[i,2], vertices_3d[j,2]], 'k-')
        
        # Add labels and title
        ax3d.set_xlabel('Proportion of Recipe A')
        ax3d.set_ylabel('Proportion of Recipe B')
        ax3d.set_zlabel('Proportion of Recipe C')
        ax3d.set_title('3D Visualization of Recipe Space\n(Probability Simplex)')
        
        # Add annotations for vertices
        ax3d.text(1, 0, 0, '100% A', color='red')
        ax3d.text(0, 1, 0, '100% B', color='red')
        ax3d.text(0, 0, 1, '100% C', color='red')
        
        # Add a point for equal mixture and annotate it
        equal_mix = np.array([1/3, 1/3, 1/3])
        ax3d.scatter([equal_mix[0]], [equal_mix[1]], [equal_mix[2]], 
                    c='green', s=100, label='Equal Mix')
        ax3d.text(1/3, 1/3, 1/3, 'Equal Mix\n(1/3 each)', color='green')
        
        # Adjust the view
        ax3d.view_init(elev=20, azim=45)
        ax3d.legend()
        
        plt.show()
        
    def _plot_coefficient_space(self, ax):
        """
        Visualize the coefficient space (probability simplex)
        """
        # Plot the triangle vertices
        vertices = np.array([[1,0,0], [0,1,0], [0,0,1]])
        ax.scatter(vertices[:,0], vertices[:,1], c='red', s=100)
        
        # Annotate vertices
        ax.annotate('100% A', (1, 0))
        ax.annotate('100% B', (0, 1))
        ax.annotate('100% C', (0, 0))
        
        # Generate points in probability simplex
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = 1 - X - Y
        
        # Plot only points where Z >= 0 (valid probability simplex)
        mask = Z >= 0
        ax.scatter(X[mask], Y[mask], c='blue', alpha=0.1)
        
        ax.set_xlabel('Proportion of Recipe A')
        ax.set_ylabel('Proportion of Recipe B')
        ax.set_title('Valid Recipe Combinations\n(Coefficient Space)')
        ax.grid(True)

# Example usage
smoothie = SmoothieRecipe()

# Create some valid recipe combinations
print("Example Recipe Combinations:")

# Equal mix of all recipes
print("\nEqual mix (1/3 each):")
print(smoothie.create_convex_combination([1/3, 1/3, 1/3]))

# Mix of just A and B
print("\n50-50 mix of Berry Blast and Creamy:")
print(smoothie.create_convex_combination([0.5, 0.5, 0]))

try:
    # This will raise an error - coefficients don't sum to 1
    print("\nTrying invalid combination (sum > 1):")
    smoothie.create_convex_combination([0.5, 0.5, 0.5])
except ValueError as e:
    print(f"Error: {e}")

try:
    # This will raise an error - negative coefficient
    print("\nTrying invalid combination (negative coefficient):")
    smoothie.create_convex_combination([1.2, -0.2, 0])
except ValueError as e:
    print(f"Error: {e}")

# Visualize the convex set
smoothie.visualize_convex_set()

"""
Key Points About Convex Sets:

1. Non-negativity and Sum-to-One Constraints:
   - All coefficients must be non-negative (≥ 0)
   - All coefficients must sum to 1 (100%)
   - These constraints ensure we stay within the convex set

2. Geometric Interpretation:
   - The convex set forms a triangle in our 2D recipe space
   - Any point inside the triangle is a valid recipe
   - Points outside cannot be reached with valid coefficients
   - Any line segment between two points in the set lies entirely within the set

3. Coefficient Space:
   - Forms a probability simplex
   - Shows all possible ways to combine base recipes
   - Each point represents a valid combination of proportions

4. Properties:
   - Closed under convex combinations
   - Contains all points that can be reached using non-negative coefficients summing to 1
   - More restrictive than affine combinations (which allow negative coefficients)

Real-world applications:
- Mixing recipes (can't use negative amounts!)
- Portfolio optimization (can't short in some cases)
- Probability distributions
- Material mixing
"""

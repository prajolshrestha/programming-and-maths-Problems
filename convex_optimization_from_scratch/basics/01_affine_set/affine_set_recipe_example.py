"""
Understanding Affine Sets: The Recipe Analogy

An affine set is a set that contains all affine combinations of its points.
Affine combination: Linear combination where coefficients sum to 1.

Recipe Analogy:
- Different recipes (points in the set) for making a dish
- New valid recipes must be combinations where proportions sum to 100%
- You can use negative proportions in math (but not in real cooking!)

Example: Imagine three basic curry recipes (points in our affine set)
- Recipe A: Spicy curry  (lots of chili, less coconut)
- Recipe B: Mild curry   (less chili, more coconut)
- Recipe C: Medium curry (balanced chili and coconut)

Any valid new recipe must be a combination where proportions = 100%
"""

import numpy as np
import matplotlib.pyplot as plt

class CurryRecipe:
    def __init__(self):
        # Define three base recipes in terms of (chili, coconut) proportions
        self.recipe_A = np.array([0.8, 0.2])  # Spicy: 80% chili, 20% coconut
        self.recipe_B = np.array([0.2, 0.8])  # Mild: 20% chili, 80% coconut
        self.recipe_C = np.array([0.5, 0.5])  # Medium: 50% chili, 50% coconut
        
        self.base_recipes = np.vstack([self.recipe_A, self.recipe_B, self.recipe_C])
        
    def create_affine_combination(self, coefficients):
        """
        Create a new recipe using affine combination
        coefficients must sum to 1
        """
        if not np.isclose(sum(coefficients), 1.0):
            raise ValueError("Coefficients must sum to 1 (100%)")
            
        return coefficients @ self.base_recipes
    
    def visualize_affine_set(self, num_points=1000):
        """
        Visualize the affine set by generating random affine combinations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot base recipes
        ax1.scatter(self.base_recipes[:, 0], self.base_recipes[:, 1], 
                   c='red', s=100, label='Base Recipes')
        ax1.annotate('Spicy (A)', (self.recipe_A[0], self.recipe_A[1]))
        ax1.annotate('Mild (B)', (self.recipe_B[0], self.recipe_B[1]))
        ax1.annotate('Medium (C)', (self.recipe_C[0], self.recipe_C[1]))
        
        # Generate both inside and outside points for affine set
        random_recipes = []
        for _ in range(num_points):
            # Generate coefficients that sum to 1 but can be negative
            coef = np.random.uniform(-1, 2, 3)  # Allow negative and >1 values
            coef = coef / coef.sum()  # Make them sum to 1
            
            new_recipe = self.create_affine_combination(coef)
            random_recipes.append(new_recipe)
            
        random_recipes = np.array(random_recipes)
        
        # Plot random affine combinations
        ax1.scatter(random_recipes[:, 0], random_recipes[:, 1], 
                   c='blue', alpha=0.1, label='Possible Recipes')
        
        ax1.set_xlabel('Chili Proportion')
        ax1.set_ylabel('Coconut Proportion')
        ax1.set_title('Affine Set of Curry Recipes\n(Including Points Outside Triangle)')
        
        # Make the plot range bigger to show outside points
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, 1.5)
        
        # Plot the coefficients triangle
        self._plot_coefficient_space(ax2)
        
        plt.tight_layout()
        
        # Now create a new figure for 3D visualization
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        # Plot base recipes in 3D
        # Using the coefficients as coordinates (they sum to 1)
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
            # Generate random coefficients that sum to 1
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
curry = CurryRecipe()

# Create some valid recipe combinations
print("Example Recipe Combinations:")

# Equal mix of all recipes
print("\nEqual mix (1/3 each):")
print(curry.create_affine_combination([1/3, 1/3, 1/3]))

# Mix of just A and B
print("\n50-50 mix of Spicy and Mild:")
print(curry.create_affine_combination([0.5, 0.5, 0]))

try:
    # This will raise an error - coefficients don't sum to 1
    print("\nTrying invalid combination:")
    curry.create_affine_combination([0.5, 0.5, 0.5])
except ValueError as e:
    print(f"Error: {e}")

# Visualize the affine set
curry.visualize_affine_set()

"""
Key Points About Affine Sets:

1. Sum-to-One Constraint:
   - All coefficients must sum to 1 (100%)
   - This ensures we stay within the affine set

2. Geometric Interpretation:
   - The affine set forms a triangle in our 2D recipe space
   - Any point inside the triangle is a valid recipe
   - Points outside cannot be reached with valid coefficients

3. Coefficient Space:
   - Forms a probability simplex
   - Shows all possible ways to combine base recipes
   - Each point represents a valid combination of proportions

4. Properties:
   - Closed under affine combinations
   - Contains all points that can be reached using coefficients summing to 1
   - More restrictive than linear combinations (which don't require sum-to-one)

Real-world applications:
- Mixing recipes
- Color blending
- Portfolio optimization
- Probability distributions
""" 
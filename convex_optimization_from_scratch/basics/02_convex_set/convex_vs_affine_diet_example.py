"""
Understanding Convex Sets vs Affine Sets: Diet Planning Analogy

Convex Set:
- Can take ANY weighted average (weights sum to 1, all weights ≥ 0)
- Like mixing meals where proportions must total 100% and can't be negative
- Example: "I can eat 70% of meal A and 30% of meal B"

Affine Set:
- Must take EXACT weighted average (weights MUST sum to 1, can be negative)
- Like following a meal plan where proportions total 100% but can "subtract" meals
- Example: "I can eat 130% of meal A and -30% of meal B"

Key Difference:
- Convex sets require non-negative weights that sum to 1
- Affine sets allow negative weights but still sum to 1
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DietPlanner:
    def __init__(self):
        # Define meals in terms of (protein, carbs, fats) in grams
        self.meal_A = np.array([30, 10, 10])  # High protein
        self.meal_B = np.array([10, 30, 10])  # High carb
        self.meal_C = np.array([10, 10, 30])  # High fat
        
        self.base_meals = np.vstack([self.meal_A, self.meal_B, self.meal_C])
    
    def create_convex_combination(self, weights):
        """
        Create a meal using convex combination
        weights must be non-negative and sum to exactly 1
        """
        weights = np.array(weights)  # Convert list to numpy array
        if np.any(weights < 0) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must be non-negative and sum to exactly 1")
        return weights @ self.base_meals
    
    def create_affine_combination(self, weights):
        """
        Create a meal using affine combination
        weights must sum to exactly 1
        """
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to exactly 1")
        return weights @ self.base_meals

    def create_cone_combination(self, weights):
        """
        Create a meal using cone combination
        weights must be non-negative (no sum constraint)
        """
        weights = np.array(weights)  # Convert list to numpy array
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        return weights @ self.base_meals

    def visualize_comparison(self, num_points=1000):
        """Compare convex, affine, and cone combinations"""
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Convex Set
        ax1 = fig.add_subplot(141, projection='3d')
        self._plot_convex_set(ax1, num_points)
        
        # 2. Affine Set
        ax2 = fig.add_subplot(142, projection='3d')
        self._plot_affine_set(ax2, num_points)
        
        # 3. Convex Cone
        ax3 = fig.add_subplot(143, projection='3d')
        self._plot_cone_set(ax3, num_points)
        
        # 4. Comparison
        ax4 = fig.add_subplot(144, projection='3d')
        self._plot_comparison(ax4, num_points)
        
        plt.tight_layout()
        plt.show()

    def _plot_convex_set(self, ax, num_points):
        """Plot convex combinations"""
        # Plot base meals
        ax.scatter(self.base_meals[:, 0], 
                  self.base_meals[:, 1], 
                  self.base_meals[:, 2], 
                  c='red', s=100, label='Base Meals')
        
        # Generate random convex combinations (non-negative weights summing to 1)
        weights = np.random.rand(num_points, 3)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]  # Normalize to sum to 1
        
        combinations = weights @ self.base_meals
        
        # Plot combinations
        scatter = ax.scatter(combinations[:, 0], 
                           combinations[:, 1], 
                           combinations[:, 2],
                           c='green',
                           alpha=0.6,
                           label='Convex Combinations')
        
        ax.set_xlabel('Protein (g)')
        ax.set_ylabel('Carbs (g)')
        ax.set_zlabel('Fats (g)')
        ax.set_title('Convex Set\n(Non-negative proportions)')
        ax.legend()

    def _plot_affine_set(self, ax, num_points):
        """Plot affine combinations"""
        # Plot base meals
        ax.scatter(self.base_meals[:, 0], 
                  self.base_meals[:, 1], 
                  self.base_meals[:, 2], 
                  c='red', s=100, label='Base Meals')
        
        # Generate random affine combinations
        weights = np.random.rand(num_points, 3)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]  # Sum exactly to 1
        
        combinations = weights @ self.base_meals
        
        # Plot combinations
        ax.scatter(combinations[:, 0], 
                  combinations[:, 1], 
                  combinations[:, 2],
                  c='blue', alpha=0.1, label='Affine Combinations')
        
        ax.set_xlabel('Protein (g)')
        ax.set_ylabel('Carbs (g)')
        ax.set_zlabel('Fats (g)')
        ax.set_title('Affine Set\n(Exact Portions)')
        ax.legend()

    def _plot_cone_set(self, ax, num_points):
        """Plot cone combinations"""
        # Plot base meals
        ax.scatter(self.base_meals[:, 0], 
                  self.base_meals[:, 1], 
                  self.base_meals[:, 2], 
                  c='red', s=100, label='Base Meals')
        
        # Generate random cone combinations (non-negative weights, no sum constraint)
        weights = np.random.rand(num_points, 3) * 2  # Scale up to show cone shape
        
        combinations = weights @ self.base_meals
        
        # Plot combinations
        scatter = ax.scatter(combinations[:, 0], 
                           combinations[:, 1], 
                           combinations[:, 2],
                           c='purple',
                           alpha=0.6,
                           label='Cone Combinations')
        
        ax.set_xlabel('Protein (g)')
        ax.set_ylabel('Carbs (g)')
        ax.set_zlabel('Fats (g)')
        ax.set_title('Convex Cone\n(Non-negative weights, no sum constraint)')
        ax.legend()

    def _plot_comparison(self, ax, num_points):
        """Plot both sets for comparison"""
        # Generate convex combinations (non-negative weights summing to 1)
        weights_convex = np.random.rand(num_points, 3)
        weights_convex = weights_convex / weights_convex.sum(axis=1)[:, np.newaxis]
        combinations_convex = weights_convex @ self.base_meals

        # Generate affine combinations (can be negative, sum to 1)
        weights_affine = np.random.randn(num_points, 3)  # Normal distribution for negative values
        weights_affine = weights_affine / weights_affine.sum(axis=1)[:, np.newaxis]
        combinations_affine = weights_affine @ self.base_meals

        # Plot both
        ax.scatter(combinations_convex[:, 0], 
                  combinations_convex[:, 1], 
                  combinations_convex[:, 2],
                  c='green', alpha=0.1, label='Convex')
        
        ax.scatter(combinations_affine[:, 0], 
                  combinations_affine[:, 1], 
                  combinations_affine[:, 2],
                  c='blue', alpha=0.1, label='Affine')

        ax.set_xlabel('Protein (g)')
        ax.set_ylabel('Carbs (g)')
        ax.set_zlabel('Fats (g)')
        ax.set_title('Comparison\nConvex (≥0) vs Affine')
        ax.legend()

# Example usage
diet = DietPlanner()

# Try some combinations
print("Example Meal Combinations:")

# Standard convex combination (non-negative weights summing to 1)
print("\nConvex combination (non-negative weights, sum to 1):")
print(diet.create_convex_combination([0.5, 0.3, 0.2]))  # Sums to 1, all non-negative

# Affine combination with negative weights
print("\nAffine combination (any weights, sum to 1):")
print(diet.create_affine_combination([1.3, -0.5, 0.2]))  # Sums to 1, can be negative

# Cone combination
print("\nCone combination (non-negative weights, no sum constraint):")
print(diet.create_cone_combination([0.8, 0.6, 0.4]))  # Just non-negative, no sum constraint

try:
    # This will raise an error - weights don't sum to 1
    print("\nTrying invalid convex combination:")
    diet.create_convex_combination([0.2, 0.15, 0.15])  # Sums to 0.5, not 1
except ValueError as e:
    print(f"Error: {e}")

# Visualize the differences
diet.visualize_comparison()

"""
Key Insights:

1. Convex Sets (Simplex) Properties:
   - Require weights to sum to exactly 1
   - All weights must be non-negative
   - Form a solid triangle including interior
   - Every point is a weighted average of vertices

2. Affine Sets Properties:
   - Require weights to sum to exactly 1
   - Weights can be negative
   - Form a plane through the points
   - Points can "extend" beyond vertices

3. Convex Cone Properties:
   - Only require weights to be non-negative
   - No constraint on sum of weights
   - Forms an infinite cone from origin
   - Can scale points by any positive number

4. Real-world Interpretation:
   Convex: "I can mix meals in any non-negative proportion totaling 100%"
   Affine: "I can use any proportions (including negative) totaling 100%"
   Cone: "I can use any non-negative amount of each meal"

5. Mathematical Properties:
   Convex: Includes all non-negative weighted averages summing to 1
   Affine: Includes all weighted averages summing to 1 (can be negative)
   Cone: Includes all non-negative weighted combinations (no sum constraint)
"""
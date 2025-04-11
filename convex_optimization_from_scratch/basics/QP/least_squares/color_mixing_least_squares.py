"""
The Color Mixing Story: Understanding Least Squares Optimization

Imagine you're a painter trying to recreate a specific color (let's call it b).
You have a set of base colors (columns of matrix A), and you need to figure out
how much of each base color to use (that's your x vector) to get as close as
possible to your target color.

Scenario 1: Unconstrained Mixing (Basic Least Squares)
--------------------------------------------------
In this version, you can:
- Use negative amounts of colors (like "removing" color)
- Use any amount of each color
- Have no limits on total paint used

Example:
    Target: A rich purple color (b)
    Base colors (A's columns): [Red, Blue, Yellow]
    Solution (x): [-0.2 Red, 0.8 Blue, 0.3 Yellow]
    
    Even though negative paint doesn't make physical sense, mathematically
    it gives us the closest match to our target color!


Scenario 2: Constrained Mixing (Constrained Least Squares)
----------------------------------------------------
Now let's add realistic painting constraints:
1. Can't use negative paint (x ≥ 0)
2. Limited total amount of paint (sum(x) ≤ max_paint)
3. Maximum amount per color (x ≤ max_per_color)

This is more realistic but might give us a worse match to our target color!
"""

import cvxpy as cp
import numpy as np

# Let's simulate both scenarios with a simple example
def color_mixing_example(with_constraints=False):
    # Setup our color space (simplified to 3D for visualization)
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

# Try both versions
unconstrained = color_mixing_example(with_constraints=False)
constrained = color_mixing_example(with_constraints=True)

print("Unconstrained Mixing:")
print(f"Mixing ratios: {unconstrained['mixing_ratios']}")
print(f"Color difference: {unconstrained['color_difference']:.4f}")
print("\nConstrained Mixing:")
print(f"Mixing ratios: {constrained['mixing_ratios']}")
print(f"Color difference: {constrained['color_difference']:.4f}")

"""
Key Insights:
1. Unconstrained mixing often gives better mathematical results but might not
   be physically realistic (negative paint?)
   
2. Adding constraints:
   - Makes the solution more realistic
   - Usually increases the color difference (worse match)
   - But gives us something we could actually create!
   
3. This is exactly like least squares optimization where:
   - A's columns are our base colors
   - b is our target color
   - x tells us how much of each base color to use
   - ||Ax - b||₂² measures how different our mixed color is from the target
""" 
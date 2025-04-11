"""
Least Squares Problem

Goal: Find values for x that minimize ||Ax - b||₂², where:
      - A is an m×n matrix (m > n, overdetermined system)
      - b is an m-dimensional vector of observations
      - x is an n-dimensional vector of parameters we want to optimize
      
      Since we have more equations (m) than unknowns (n), an exact solution usually doesn't exist.
      Instead, we minimize the sum of squared differences between Ax and b.

Intuition:
    - A: Design/measurement matrix that defines the linear system
    - b: Observed data/measurements
    - x: Parameters/coefficients we're trying to estimate
    - ||Ax - b||₂²: Sum of squared residuals (difference between predictions and observations)

Mathematical formulation:
    minimize ||Ax - b||₂²
    where ||·||₂ denotes the L2 (Euclidean) norm

The residual norm ||Ax - b||₂ quantifies how well our solution fits the data:
    - Smaller values indicate better fits
    - Zero would indicate a perfect fit (rarely achievable in overdetermined systems)

Geometric interpretation:
    - Think of b as a point in m-dimensional space
    - The columns of A span a n-dimensional subspace (n < m)
    - Least squares finds the point in this subspace closest to b
    - This is equivalent to projecting b onto the subspace
    - The projection is orthogonal (perpendicular) to minimize distance
    - That's why it's called "orthogonal projection"!

    "We are projecting b onto the subspace spanned by A's columns, 
    and x gives us the coefficients that define this projection"

For example, if we write out A's columns:
A = [a₁ a₂ a₃ ... a₁₅] where each aᵢ is a 20-dimensional column vector
x = [x₁ x₂ x₃ ... x₁₅]ᵀ are the coefficients
A@x = x₁a₁ + x₂a₂ + x₃a₃ + ... + x₁₅a₁₅

So:
A's columns define the "basis vectors" of our subspace
x tells us how much of each basis vector to use
A@x is the resulting linear combination

Think of it like mixing colors:
    Target color: b (mixed color)

    A's columns would be like your basic colors (red, blue, yellow) (available 3 base colors)
    x would be how much of each base color to use (3 different mixing ratios)
    A@x would be the final mixed color (best attempt to recreate the target color using our base colors)


This is unconstrained version: We can use negative amounts or any amount of base colors

With constraints we could enforce:
    Only positive amounts of colors (x >= 0)
    Total amount of paint limited (cp.sum(x) <= max_paint)
    Maximum amount for each color (x <= max_per_color)

The unconstrained version often has a closed-form solution, but adding constraints typically requires numerical optimization methods.
"""

import cvxpy as cp
import numpy as np

# Generate data.
m = 20 # 20 equations
n = 15 # 15 unknowns (parameters)
np.random.seed(1)

# Define our system 
# This is an overdetermined system (more equations than unknowns)
#We can't usually get A@x = b exactly (b is in 20D space, but A@x can only reach a 15D subspace)
A = np.random.randn(m, n) # 20 x 15 matrix | A is our 20×15 matrix that transforms x into the same space as b
b = np.random.randn(m) # 20 x 1 vector


# Define and solve the CVXPY problem
x = cp.Variable(n)
cost = cp.sum_squares(A@x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

print(f"The optimal value is {prob.value}")
print(f"The optimal x is {x.value}")
print(f"The norm of the residual is {cp.norm(A@x - b, p=2).value}")

"""
Current unconstrained version: We can use negative amounts or any amount of base colors
With constraints we could enforce:
Only positive amounts of colors (x >= 0)
Total amount of paint limited (cp.sum(x) <= max_paint)
Maximum amount for each color (x <= max_per_color)
The unconstrained version often has a closed-form solution, but adding constraints typically requires numerical optimization methods.
"""
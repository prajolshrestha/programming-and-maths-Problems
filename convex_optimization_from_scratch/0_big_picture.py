import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Generate x values
x = np.linspace(-3, 3, 1000)

# Convex function (parabola)
y_convex = x**2

# Non-convex function with local minimum at origin (higher) and global minimum to the right (lower)
y_nonconvex = 0.5 * (x - 0.5)**2 + 1.5 * np.exp(-2 * x**2)

# Plot convex function
ax1.plot(x, y_convex, 'r-', linewidth=2)
#ax1.set_title('Convex', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-0.5, 4)

# Mark the global minimum for convex function
min_idx_convex = np.argmin(y_convex)
ax1.plot(x[min_idx_convex], y_convex[min_idx_convex], 'bo', markersize=8)

# Plot non-convex function
ax2.plot(x, y_nonconvex, 'r-', linewidth=2)
#ax2.set_title('Non-convex', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-0.5, 4)

# Find local minima for non-convex function
local_min_indices = argrelextrema(y_nonconvex, np.less, order=20)[0]

# Filter significant local minima within range
significant_minima = []
for idx in local_min_indices:
    if -2.8 <= x[idx] <= 2.8:
        significant_minima.append((idx, y_nonconvex[idx]))

# Sort by y-value to identify global vs local minima
if significant_minima:
    significant_minima.sort(key=lambda x: x[1])
    
    # First mark local minima (higher values)
    for idx, y_val in significant_minima[1:]:
        ax2.plot(x[idx], y_nonconvex[idx], 'go', markersize=6)
    
    # Then mark global minimum (lowest point) - this will appear on top
    global_min_idx = significant_minima[0][0]
    ax2.plot(x[global_min_idx], y_nonconvex[global_min_idx], 'bo', markersize=8)

# Remove axis ticks for cleaner look
for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
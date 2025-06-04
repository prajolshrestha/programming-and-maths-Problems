import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Create custom colormap: blue (low/minima) to red (high/maxima)
colors = ['#0000FF', '#4169E1', '#00CED1', '#90EE90', '#FFFF00', '#FFA500', '#FF0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_to_red', colors, N=n_bins)

# Define grid for plotting
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# =============================================================================
# LEFT PLOT: CONVEX OBJECTIVE AND CONVEX CONSTRAINTS
# =============================================================================

# Convex objective function (paraboloid for minimization)
# Lower values (blue) at center, higher values (red) at edges
Z_convex = 0.5 * (X**2 + Y**2) + 1

# Plot contour filled plot for convex function
contour1 = ax1.contourf(X, Y, Z_convex, levels=20, cmap=cmap, alpha=0.8)
contour_lines1 = ax1.contour(X, Y, Z_convex, levels=15, colors='white', 
                            alpha=0.6, linewidths=1)

# Create convex constraint region (circular constraint)
theta = np.linspace(0, 2*np.pi, 100)
constraint_radius = 2.5
x_constraint = constraint_radius * np.cos(theta)
y_constraint = constraint_radius * np.sin(theta)

# Plot constraint boundary
ax1.plot(x_constraint, y_constraint, 'k-', linewidth=4, alpha=1.0, 
         label='Feasible boundary')

# Fill the feasible region with a subtle overlay
circle = patches.Circle((0, 0), constraint_radius, fill=False, 
                       edgecolor='black', linewidth=4, alpha=0.8)
ax1.add_patch(circle)

# Mark the optimal point (minimum at center - should appear blue)
optimal_x, optimal_y = 0, 0  # For convex case, optimum is at center
ax1.scatter([optimal_x], [optimal_y], color='darkblue', s=150, alpha=1, 
           marker='*', edgecolor='black', linewidth=2, 
           label='Global minimum', zorder=5)

# Styling for left plot
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
#ax1.set_title('Convex Objective and Convex Constraints', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Add colorbar for left plot
cbar1 = plt.colorbar(contour1, ax=ax1, shrink=0.8)
cbar1.set_label('Objective Value f(x,y)', fontsize=10)

# =============================================================================
# RIGHT PLOT: NONCONVEX OBJECTIVE AND NONCONVEX CONSTRAINTS
# =============================================================================

# Non-convex objective function (multiple local maxima and minima)
# Adjust to have clear minima (blue) and maxima (red)
Z_nonconvex = (np.sin(X*2) * np.cos(Y*2) * 3 + 
               np.sin(X*Y) * 2 + 
               0.1 * (X**2 + Y**2) + 3)

# Plot contour filled plot for non-convex function
contour2 = ax2.contourf(X, Y, Z_nonconvex, levels=25, cmap=cmap, alpha=0.8)
contour_lines2 = ax2.contour(X, Y, Z_nonconvex, levels=20, colors='white', 
                            alpha=0.6, linewidths=1)

# Create non-convex constraint with multiple elliptical holes (outer boundary)
t = np.linspace(0, 2*np.pi, 200)
# Create a more complex non-convex constraint shape
x_nonconvex_constraint = 2.2 * np.cos(t) + 0.2 * np.cos(5*t)
y_nonconvex_constraint = 2.0 * np.sin(t) + 0.3 * np.sin(3*t)

# Plot main constraint boundary
ax2.plot(x_nonconvex_constraint, y_nonconvex_constraint, 'k-', 
         linewidth=4, alpha=1.0, label='Feasible boundary')

# Create multiple elliptical holes in the main constraint region (all infeasible)
# All holes are ellipses with same size
a, b = 0.4, 0.25  # semi-major and semi-minor axes (same for all)

hole_centers = [(0.8, 0.3), (-0.9, -0.5), (-0.2, 0.9), (0.3, -0.8), (-1.2, 0.4)]

# Plot all elliptical holes (infeasible regions inside the main constraint)
for i, (center_x, center_y) in enumerate(hole_centers):
    # Create elliptical hole
    t_hole = np.linspace(0, 2*np.pi, 100)
    x_hole = a * np.cos(t_hole) + center_x
    y_hole = b * np.sin(t_hole) + center_y
    
    # Plot hole boundary
    ax2.plot(x_hole, y_hole, 'gray', linewidth=3, alpha=1.0)
    
    # Fill the hole with gray to show it's infeasible
    ellipse = patches.Ellipse((center_x, center_y), 2*a, 2*b, 
                             fill=True, facecolor='lightgray', alpha=0.7, 
                             edgecolor='gray', linewidth=3)
    ax2.add_patch(ellipse)

# Add another non-convex region (separate island)
t2 = np.linspace(0, 2*np.pi, 100)
x_constraint2 = 0.6 * np.cos(t2) - 2.2
y_constraint2 = 0.6 * np.sin(t2) + 1.6

# Plot the separate island
#ax2.plot(x_constraint2, y_constraint2, 'k-', linewidth=4, alpha=1.0)

# Mark multiple local optima for non-convex case
# Blue stars for minima, red stars for maxima
# local_minima_x = [-1.5, 1.2]
# local_minima_y = [0.8, -1.1]
# local_maxima_x = [0.8, -0.5]
# local_maxima_y = [1.5, -1.8]

# # Plot local minima (blue)
# for i, (opt_x, opt_y) in enumerate(zip(local_minima_x, local_minima_y)):
#     if (-3 <= opt_x <= 3) and (-3 <= opt_y <= 3):
#         ax2.scatter([opt_x], [opt_y], color='darkblue', s=120, alpha=1, 
#                    marker='*', edgecolor='black', linewidth=2, zorder=5)

# # Plot local maxima (red)
# for i, (opt_x, opt_y) in enumerate(zip(local_maxima_x, local_maxima_y)):
#     if (-3 <= opt_x <= 3) and (-3 <= opt_y <= 3):
#         ax2.scatter([opt_x], [opt_y], color='darkred', s=120, alpha=1, 
#                    marker='*', edgecolor='black', linewidth=2, zorder=5)

# Add text to show these are local optima
# ax2.text(-1.5, 1.2, 'Local\nMinima', fontsize=10, ha='center', va='bottom',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
# ax2.text(0.8, 1.8, 'Local\nMaxima', fontsize=10, ha='center', va='bottom',
#          bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.7))

# Styling for right plot
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
#ax2.set_title('Nonconvex Objective and Nonconvex Constraints', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Add colorbar for right plot
cbar2 = plt.colorbar(contour2, ax=ax2, shrink=0.8)
cbar2.set_label('Objective Value f(x,y)', fontsize=10)

# # Create custom legend
legend_elements = [
    plt.Line2D([0], [0], color='black', lw=4, label='Feasible boundary'),
    plt.Line2D([0], [0], color='gray', lw=3, label='Infeasible regions')#,
#     plt.Line2D([0], [0], marker='*', color='darkblue', lw=0, markersize=12, 
#                markeredgecolor='black', label='Local minima'),
#     plt.Line2D([0], [0], marker='*', color='darkred', lw=0, markersize=12, 
#                markeredgecolor='black', label='Local maxima')
]

# # Add legends
# ax1.legend([plt.Line2D([0], [0], color='black', lw=4),
#            plt.Line2D([0], [0], marker='*', color='darkblue', lw=0, markersize=12)],
#           ['Feasible boundary', 'Global minimum'], loc='upper right')

ax2.legend(legend_elements, ['Feasible boundary', 'Infeasible regions'], 
          loc='upper right')

# Adjust layout and save figure BEFORE showing
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)

# Save the figure (must be done before plt.show())
plt.savefig('convex_vs_nonconvex_2d_optimization.png', dpi=300, bbox_inches='tight')
print("2D Figure saved as 'convex_vs_nonconvex_2d_optimization.png'")

# Display the figure
plt.show()

print("2D Visualization complete!")
print("\nKey differences illustrated:")
print("1. LEFT (Convex): Smooth contour levels with circular constraint")
print("   - Single global minimum (blue star) - BLUE regions show low values")
print("   - Convex feasible region (circle)")
print("   - Concentric contour lines - easy optimization")
print("\n2. RIGHT (Non-convex): Complex contour patterns with holey constraints")
print("   - Multiple local minima (blue stars) and maxima (red stars)")
print("   - Non-convex feasible regions with 5 identical elliptical holes")
print("   - All holes have same size: semi-major axis = 0.4, semi-minor axis = 0.25")
print("   - Disconnected feasible island")
print("   - Irregular contour lines - challenging optimization!")
print("   - BLUE regions = low values (minima), RED regions = high values (maxima)")
print("   - Gray filled ellipses show infeasible holes")
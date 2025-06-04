import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Set up the figure with two subplots side by side
fig = plt.figure(figsize=(16, 8))

# Create custom colormap for the 3D surfaces (blue to red: minimum = blue, maximum = red)
colors = ['#0000FF', '#4169E1', '#00CED1', '#FFD700', '#FFA500', '#FF0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Define grid for plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# =============================================================================
# LEFT PLOT: CONVEX OBJECTIVE AND CONVEX CONSTRAINTS
# =============================================================================
ax1 = fig.add_subplot(121, projection='3d')

# Convex objective function (paraboloid)
Z_convex = 0.5 * (X**2 + Y**2) + 1

# Plot the convex surface
surf1 = ax1.plot_surface(X, Y, Z_convex, cmap=cmap, alpha=0.8, 
                        linewidth=0, antialiased=True)

# Create convex constraint region (circular constraint)
theta = np.linspace(0, 2*np.pi, 100)
constraint_radius = 2.5
x_constraint = constraint_radius * np.cos(theta)
y_constraint = constraint_radius * np.sin(theta)

# Plot constraint boundary on the base
ax1.plot(x_constraint, y_constraint, 0, 'k-', linewidth=3, alpha=0.8)

# Fill the feasible region
theta_fill = np.linspace(0, 2*np.pi, 50)
r_fill = np.linspace(0, constraint_radius, 20)
R_fill, Theta_fill = np.meshgrid(r_fill, theta_fill)
X_fill = R_fill * np.cos(Theta_fill)
Y_fill = R_fill * np.sin(Theta_fill)

# Create contour lines on the base plane
contours1 = ax1.contour(X, Y, Z_convex, levels=10, colors='lightblue', 
                       alpha=0.6, linewidths=1, zdir='z', offset=0)

# Mark the optimal point
optimal_x, optimal_y = 0, 0  # For convex case, optimum is at center
ax1.scatter([optimal_x], [optimal_y], [0.5 * (optimal_x**2 + optimal_y**2) + 1], 
           color='white', s=100, alpha=1, edgecolors='black', linewidth=2)

# Styling for left plot
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_zlim(0, 10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y)')
#ax1.set_title('Convex Objective and Convex Constraints', fontsize=14, fontweight='bold')
ax1.view_init(elev=25, azim=45)

# =============================================================================
# RIGHT PLOT: NONCONVEX OBJECTIVE AND NONCONVEX CONSTRAINTS
# =============================================================================
ax2 = fig.add_subplot(122, projection='3d')

# Non-convex objective function (multiple local minima)
Z_nonconvex = (np.sin(X*2) * np.cos(Y*2) * 2 + 
               0.3 * (X**2 + Y**2) + 
               np.sin(X*Y) * 1.5 + 4)

# Plot the non-convex surface
surf2 = ax2.plot_surface(X, Y, Z_nonconvex, cmap=cmap, alpha=0.8, 
                        linewidth=0, antialiased=True)

# Create non-convex constraint with multiple elliptical holes (outer boundary)
t = np.linspace(0, 2*np.pi, 200)
# Create a more complex non-convex constraint shape
x_nonconvex_constraint = 2.2 * np.cos(t) + 0.2 * np.cos(5*t)
y_nonconvex_constraint = 2.0 * np.sin(t) + 0.3 * np.sin(3*t)

# Create multiple elliptical holes in the main constraint region (all infeasible)
holes_data = []

# All holes are ellipses with same size
a, b = 0.4, 0.25  # semi-major and semi-minor axes (same for all)

# Hole 1 - elliptical
t_hole1 = np.linspace(0, 2*np.pi, 100)
x_hole1 = a * np.cos(t_hole1) + 0.8
y_hole1 = b * np.sin(t_hole1) + 0.3
holes_data.append((x_hole1, y_hole1, a, 0.8, 0.3))

# Hole 2 - elliptical
t_hole2 = np.linspace(0, 2*np.pi, 100)
x_hole2 = a * np.cos(t_hole2) - 0.9
y_hole2 = b * np.sin(t_hole2) - 0.5
holes_data.append((x_hole2, y_hole2, a, -0.9, -0.5))

# Hole 3 - elliptical
t_hole3 = np.linspace(0, 2*np.pi, 100)
x_hole3 = a * np.cos(t_hole3) - 0.2
y_hole3 = b * np.sin(t_hole3) + 0.9
holes_data.append((x_hole3, y_hole3, a, -0.2, 0.9))

# Hole 4 - elliptical
t_hole4 = np.linspace(0, 2*np.pi, 100)
x_hole4 = a * np.cos(t_hole4) + 0.3
y_hole4 = b * np.sin(t_hole4) - 0.8
holes_data.append((x_hole4, y_hole4, a, 0.3, -0.8))

# Hole 5 - elliptical
t_hole5 = np.linspace(0, 2*np.pi, 100)
x_hole5 = a * np.cos(t_hole5) - 1.2
y_hole5 = b * np.sin(t_hole5) + 0.4
holes_data.append((x_hole5, y_hole5, a, -1.2, 0.4))

# Add another non-convex region (separate island)
t2 = np.linspace(0, 2*np.pi, 100)
x_constraint2 = 0.6 * np.cos(t2) - 2.2
y_constraint2 = 0.6 * np.sin(t2) + 1.6

# Plot non-convex constraint boundaries
ax2.plot(x_nonconvex_constraint, y_nonconvex_constraint, 0, 'k-', 
         linewidth=3, alpha=0.8, label='Feasible boundary')

# Plot all elliptical holes (infeasible regions inside the main constraint)
for i, (x_hole, y_hole, radius, center_x, center_y) in enumerate(holes_data):
    ax2.plot(x_hole, y_hole, 0, 'k-', 
             linewidth=2.5, alpha=0.8)
    
    # Fill each elliptical hole area to make it more visible
    theta_hole = np.linspace(0, 2*np.pi, 30)
    u = np.linspace(0, 1, 8)
    U, Theta_hole = np.meshgrid(u, theta_hole)
    X_hole = U * a * np.cos(Theta_hole) + center_x
    Y_hole = U * b * np.sin(Theta_hole) + center_y
    Z_hole = np.ones_like(X_hole) * 0.1
    ax2.plot_surface(X_hole, Y_hole, Z_hole, color='red', alpha=0.2)

# Plot the separate island
#ax2.plot(x_constraint2, y_constraint2, 0, 'k-', linewidth=3, alpha=0.8)

# Create contour lines on the base plane for non-convex case
contours2 = ax2.contour(X, Y, Z_nonconvex, levels=15, colors='lightblue', 
                       alpha=0.6, linewidths=1, zdir='z', offset=0)

# Mark multiple local optima for non-convex case
local_optima_x = [-1.5, 1.2, 0.8, -0.5]
local_optima_y = [0.8, -1.1, 1.5, -1.8]
for opt_x, opt_y in zip(local_optima_x, local_optima_y):
    if (-3 <= opt_x <= 3) and (-3 <= opt_y <= 3):
        opt_z = (np.sin(opt_x*2) * np.cos(opt_y*2) * 2 + 
                0.3 * (opt_x**2 + opt_y**2) + 
                np.sin(opt_x*opt_y) * 1.5 + 4)
        ax2.scatter([opt_x], [opt_y], [opt_z], color='white', s=80, alpha=1, 
                   edgecolors='black', linewidth=2)

# Styling for right plot
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_zlim(0, 10)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('f(x,y)')
#ax2.set_title('Nonconvex Objective and Nonconvex Constraints', fontsize=14, fontweight='bold')
ax2.view_init(elev=25, azim=45)

# Adjust layout and save figure BEFORE showing
plt.tight_layout()
plt.subplots_adjust(wspace=0.1)

# Save the figure (must be done before plt.show())
plt.savefig('convex_vs_nonconvex_3d_optimization.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'convex_vs_nonconvex_3d_optimization.png'")

# Display the figure
plt.show()

print("Visualization complete!")
print("\nKey differences illustrated:")
print("1. LEFT (Convex): Smooth bowl-shaped objective with circular constraint")
print("   - Single global minimum (white dot with black edge)")
print("   - Convex feasible region")
print("   - Optimization is straightforward")
print("   - Blue = low values (minimum), Red = high values (maximum)")
print("\n2. RIGHT (Non-convex): Wavy objective with complex constraint shape")
print("   - Multiple local minima (white dots with black edges)")
print("   - Non-convex feasible regions with 5 identical elliptical holes (red boundaries)")
print("   - All holes have same size: semi-major axis = 0.4, semi-minor axis = 0.25")
print("   - Disconnected feasible island")
print("   - Optimization is extremely challenging with uniform elliptical obstacles!")
print("   - Red boundaries show all identical infeasible elliptical holes")
print("   - Blue = low values (minimum), Red = high values (maximum)")
import numpy as np
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Generate data for the cost function curve
w = np.linspace(-2, 4, 1000)
J_w = 0.5 * (w - 1.5)**2 + 0.3  # Parabolic cost function with minimum at w=1.5

# Plot the main cost function curve
ax.plot(w, J_w, color='#4A90E2', linewidth=4, label='Cost Function J(w)')

# Define key points
w_initial = 3.2  # Initial weight position
w_min = 1.5      # Global minimum position
J_initial = 0.5 * (w_initial - 1.5)**2 + 0.3
J_min = 0.3

# Plot the initial weight point
ax.plot(w_initial, J_initial, 'ko', markersize=12, zorder=5)

# Plot the global minimum point
ax.plot(w_min, J_min, 'ko', markersize=8, zorder=5)

# Calculate gradient at initial point (derivative of the cost function)
gradient_slope = w_initial - 1.5  # Derivative of 0.5*(w-1.5)^2
gradient_length = 0.8

# Draw gradient arrow (tangent line showing gradient direction)
gradient_start_w = w_initial - 0.3
gradient_end_w = w_initial + 0.3
gradient_start_J = J_initial - 0.3 * gradient_slope
gradient_end_J = J_initial + 0.3 * gradient_slope

ax.plot([gradient_start_w, gradient_end_w], [gradient_start_J, gradient_end_J], 
        'k-', linewidth=2, zorder=4)

# Add gradient arrow
ax.annotate('', xy=(gradient_end_w, gradient_end_J), 
            xytext=(gradient_start_w, gradient_start_J),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Draw dashed vertical line from initial point
ax.plot([w_initial, w_initial], [0, J_initial], 'k--', linewidth=2, alpha=0.7)

# Draw arrows showing the optimization path
arrow_positions = [
    (w_initial - 0.2, J_initial - 0.15),
    (w_initial - 0.5, J_initial - 0.4),
    (w_initial - 0.8, J_initial - 0.7),
    (w_initial - 1.1, J_initial - 1.0)
]

for i, (w_pos, J_pos) in enumerate(arrow_positions):
    ax.annotate('', xy=(w_pos - 0.15, J_pos - 0.1), xytext=(w_pos, J_pos),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Add labels and annotations
ax.text(w_initial + 0.1, J_initial + 0.3, 'Initial\nweight', fontsize=14, 
        ha='left', va='bottom', fontweight='bold')

ax.text(w_initial + 0.4, J_initial - 0.2, 'Gradient', fontsize=12, 
        ha='left', va='center', rotation=25)

ax.text(w_min - 0.3, J_min - 0.4, f'Global cost minimum\nJ_min(w)', fontsize=12, 
        ha='center', va='top', fontweight='bold')

# Set labels and title
ax.set_xlabel('w', fontsize=16, fontweight='bold')
ax.set_ylabel('J(w)', fontsize=16, fontweight='bold', rotation=0, labelpad=20)

# Customize the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -2))
ax.spines['bottom'].set_position(('data', 0))

# Add arrows to axes
ax.annotate('', xy=(4.2, 0), xytext=(4, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(-2, 3), xytext=(-2, 2.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Set axis limits
ax.set_xlim(-2.2, 4.2)
ax.set_ylim(-0.2, 3)

# Remove tick marks but keep the grid subtle
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, alpha=0.2)

# Adjust layout and save
plt.tight_layout()

# Save the figure
plt.savefig('gradient_descent_visualization.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Also save as PDF for vector graphics
plt.savefig('gradient_descent_visualization.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Gradient descent visualization saved as:")
print("- gradient_descent_visualization.png")
print("- gradient_descent_visualization.pdf")

plt.show()
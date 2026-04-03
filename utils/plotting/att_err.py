import matplotlib.pyplot as plt
import numpy as np

# 1. Define the attitude error levels (in degrees)
error_levels = [5, 15, 30, 60, 90]

# 2. Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Set colors for the circles (optional, can be customized)
circle_colors = ['red', 'blue', 'green', 'orange', 'purple']  # Corresponding to the error levels defined above

# 3. Draw the concentric circles
for error in error_levels:
    # Create the circle patch
    circle = plt.Circle((0, 0), error, fill=False, linestyle='--', color=circle_colors[error_levels.index(error)], linewidth=1.5)
    ax.add_patch(circle)

    # Generate 100 points drawn from a uniform distribution within the circle to visualize the error distribution
    num_points = 100
    points = []
    while len(points) < num_points:
        x, y = np.random.uniform(-error, error, 2)
        if x**2 + y**2 <= error**2:  # Check if the point is within the circle
            points.append((x, y))
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], color=circle_colors[error_levels.index(error)], alpha=0.5)
    
    # Add a label to each circle 
    # (Placed at the top of each circle with a white background for readability)
    ax.text(0, error, f' {error}° ', color='midnightblue', 
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.8))

# 4. Plot the center point (Boresight / 0 error)
ax.plot(0, 0, marker='+', color='black', markersize=12, label='V-bar Attitude (0° Error)')

# 5. Format the axes
max_val = max(error_levels) + 10 # Add some padding to the edges
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)

# CRITICAL: Ensure the aspect ratio is equal so circles are perfectly round, not ellipses
ax.set_aspect('equal')

# 6. Add labels, title, and grid
ax.set_xlabel('Pointing Error X (degrees)')
ax.set_ylabel('Pointing Error Y (degrees)')
ax.set_title('Attitude Error Boundaries')
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
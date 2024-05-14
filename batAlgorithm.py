#import the required libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the Bat Class: Create a class to represent bats with attributes like position, velocity, frequency, and loudness.
class Bat:
    def __init__(self, position, velocity, frequency, loudness):
        self.position = position
        self.velocity = velocity
        self.frequency = frequency
        self.loudness = loudness

# Define the Objective Function: Implement the objective function you want to optimize. Here, we'll use the Rastrigin function as an example.

def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# Initialize bats
num_bats = 10
dimension = 2

bats = [Bat(np.random.uniform(-5, 5, dimension), np.zeros(dimension), np.zeros(dimension), 0) for _ in range(num_bats)]

# Define Bat Algorithm Parameters
max_iterations = 1000

# Initialize best solution
best_solution = None
best_fitness = float('inf')

# Optimization loop
for iteration in range(max_iterations):
    for bat in bats:
        # Update bat positions
        bat.position += bat.velocity
        
        # Evaluate fitness
        fitness = rastrigin(*bat.position)
        
        # Update best solution
        if fitness < best_fitness:
            best_solution = bat.position
            best_fitness = fitness
        
        # Update velocity
        bat.velocity += np.random.uniform(-1, 1, dimension) * (best_solution - bat.position)
        
        # Echolocation
        if np.random.rand() > bat.loudness:
            bat.position = best_solution + np.random.uniform(-1, 1, dimension) * np.random.uniform(-5, 5)
        
        # Update loudness and frequency
        bat.loudness *= 0.9
        bat.frequency *= 1.1

# Output best solution
print("Best solution found:", best_solution)
print("Fitness:", best_fitness)

# Plot the optimization process
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for visualization
x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z = rastrigin(x, y)

# Plot the 3D surface of the objective function
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

# Scatter plot the bats' positions
for bat in bats:
    ax.scatter(bat.position[0], bat.position[1], rastrigin(*bat.position), c='r', marker='o')

# Show plot
plt.show()

# Print the optimal solution found
print("Optimal solution found at (x, y) =", best_solution)

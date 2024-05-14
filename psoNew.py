import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the fitness function you want to optimize
def fitness_function(x, y, z):
    return x**2 + y**2 + z**2   # Example: Minimize the sum of squares

# Define PSO parameters
num_particles = 100
num_dimensions = 3
num_iterations = 10
c1 = 1.5  # Cognitive parameter
c2 = 1.5  # Social parameter
w = 0.5  # Inertia weight
v_max = 1.0  # Maximum velocity

# Initialize particles with random positions and velocities
particles = []
for _ in range(num_particles):
    position = np.random.uniform(-10, 10, num_dimensions)  # Random initial positions within a range
    velocity = np.random.uniform(-v_max, v_max, num_dimensions)  # Random initial velocities
    particles.append([position, velocity, position, fitness_function(*position)])  # [position, velocity, best_position, best_fitness]


# Initialize the global best position and fitness value
global_best_position = None
global_best_fitness = float('inf')  # + infinity as max


# PSO optimization loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        position, velocity, best_position, best_fitness = particles[i]

        # Evaluate the fitness of the current position
        current_fitness = fitness_function(*position)
     
        # print(current_fitness)
        # print(best_fitness)
        
        # Update personal best
        if current_fitness < best_fitness:
            best_position = position
            best_fitness = current_fitness

        # Update global best
        if current_fitness < global_best_fitness:
            global_best_position = position
            global_best_fitness = current_fitness

        # Update velocity and position
        new_velocity = (w * velocity +
                        c1 * np.random.random(num_dimensions) * (best_position - position) +
                        c2 * np.random.random(num_dimensions) * (global_best_position - position))
                
        # Ensure velocity does not exceed the maximum
        new_velocity = np.minimum(v_max, np.maximum(-v_max, new_velocity))

        new_position = position + new_velocity
        particles[i] = [new_position, new_velocity, best_position, best_fitness]


# Plot the optimization process
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for visualization
x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z = x**2 + y**2

# Plot the 3D surface of the objective function
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

# Extract the x, y, and z coordinates of the best position found
best_x, best_y, best_z = global_best_position

# Scatter plot the particles' best positions
best_positions = np.array([particle[2] for particle in particles])
best_fitnesses = np.array([particle[3] for particle in particles])
ax.scatter(best_positions[:, 0], best_positions[:, 1], best_fitnesses[:, ], c='r', marker='o', label='Best Particles')

# Scatter plot the global best position
ax.scatter(best_x, best_y, best_z, c='g', s=200, marker='*', label='Global Best')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('PSO Optimization')
plt.legend()
plt.show()

# Print the optimal solution found
print("Optimal solution found at (x, y, z) =", global_best_position)
import random

class Particle:
    def __init__(self, dim):
        self.position = [random.random() for _ in range(dim)]
        self.velocity = [random.random() for _ in range(dim)]
        self.best_position = self.position[:]
        self.best_fitness = float('inf')

def objective_function(x):
    # Replace this with your objective function
    return sum(xi**2 for xi in x)

def update_velocity(particle, global_best_position, inertia_weight, c1, c2):
    r1, r2 = random.random(), random.random()
    inertia_term = [inertia_weight * vi for vi in particle.velocity]
    cognitive_term = [c1 * r1 * (bi - xi) for xi, bi in zip(particle.best_position, particle.position)]
    social_term = [c2 * r2 * (bg - xi) for xi, bg in zip(global_best_position, particle.position)]
    new_velocity = [v + c + s for v, c, s in zip(inertia_term, cognitive_term, social_term)]
    return new_velocity

def particle_swarm_optimization(dim, num_particles, num_iterations, c1, c2, inertia_weight):
    particles = [Particle(dim) for _ in range(num_particles)]

    global_best_position = None
    global_best_fitness = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            fitness = objective_function(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]

            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position[:]

        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position, inertia_weight, c1, c2)
            particle.position = [xi + vi for xi, vi in zip(particle.position, particle.velocity)]

    return global_best_position, global_best_fitness

if __name__ == "__main__":
    # Parameters
    dim = 10  # Dimensionality of the problem
    num_particles = 20  # Number of particles
    num_iterations = 100  # Number of iterations
    c1 = 2.0  # Cognitive parameter
    c2 = 2.0  # Social parameter
    inertia_weight = 0.9  # Inertia weight

    # Run PSO algorithm
    best_position, best_fitness = particle_swarm_optimization(dim, num_particles, num_iterations, c1, c2, inertia_weight)

    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)

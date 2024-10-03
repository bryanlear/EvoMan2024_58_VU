import numpy as np
import os
import sys
import pandas as pd 
from evoman.environment import Environment
from controller import player_controller
import time

# Set up environment (same as demo script)
experiment_name = 'pso_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# EvoMan for PSO
env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# PSO parameters
n_particles = 100   # Particles (population size)
n_iterations = 30  # Iterations (generations)
w = 0.6            # Inertia weight
c1 = 1        # Cognitive (influence personal best)
c2 = 2         # Social (influence global best (swarm))

# NN parameter size
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Random particle positions and velocities
particles = np.random.uniform(-1, 1, (n_particles, n_vars))
velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_vars))

# Personal and global best positions
personal_best_positions = np.copy(particles)
personal_best_fitness = np.full(n_particles, -np.inf)
global_best_position = None
global_best_fitness = -np.inf

# To store statistics for all generations
mean_fitness_list = []
std_fitness_list = []
max_fitness_list = []

# Fitness function of particle (plays game with given weights)
def evaluate(particle):
    return np.array(list(map(lambda w: env.play(pcont=w)[0], particle)))

# Start first generation
fitness_values = evaluate(particles)

# Update personal / global best 
for i in range(n_particles):
    if fitness_values[i] > personal_best_fitness[i]:
        personal_best_fitness[i] = fitness_values[i]
        personal_best_positions[i] = particles[i]
    if fitness_values[i] > global_best_fitness:
        global_best_fitness = fitness_values[i]
        global_best_position = particles[i]

# PSO main loop
for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations} - Global best fitness: {global_best_fitness}")

    # Update particle velocities / positions
    for i in range(n_particles):
        # Calculate new velocity
        inertia = w * velocities[i]
        cognitive = c1 * np.random.uniform(0, 1, n_vars) * (personal_best_positions[i] - particles[i])
        social = c2 * np.random.uniform(0, 1, n_vars) * (global_best_position - particles[i])
        velocities[i] = inertia + cognitive + social

        # Update position
        particles[i] = particles[i] + velocities[i]

        # Set position within bounds [-1, 1]
        particles[i] = np.clip(particles[i], -1, 1)

    # Evaluate new fitness values
    fitness_values = evaluate(particles)

    # Update personal / global bests
    for i in range(n_particles):
        if fitness_values[i] > personal_best_fitness[i]:
            personal_best_fitness[i] = fitness_values[i]
            personal_best_positions[i] = particles[i]
        if fitness_values[i] > global_best_fitness:
            global_best_fitness = fitness_values[i]
            global_best_position = particles[i]
    
    # Calculate mean, std dev, and max fitness for current generation
    mean_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    max_fitness = np.max(fitness_values)

    # Store values for later analysis
    mean_fitness_list.append(mean_fitness)
    std_fitness_list.append(std_fitness)
    max_fitness_list.append(max_fitness)

    # Print mean, std dev, and max fitness for current generation
    print(f"Iteration {iteration + 1}/{n_iterations} - Mean fitness: {mean_fitness}, Std Dev: {std_fitness}, Max fitness: {max_fitness}")

    # Save best solution so far
    np.savetxt(f"{experiment_name}/best_particle.txt", global_best_position)

print("PSO optimization complete.")
print(f"Best solution found with fitness: {global_best_fitness}")

# Save mean, std dev, and max fitness values to a CSV file for this run
results_df = pd.DataFrame({
    'Generation': np.arange(1, n_iterations + 1),
    'MeanFitness': mean_fitness_list,
    'StdDevFitness': std_fitness_list,
    'MaxFitness': max_fitness_list
})

# Save results to CSV
results_file_path = f"{experiment_name}/run_{time.time()}_results.csv"
results_df.to_csv(results_file_path, index=False)
print(f"Results saved to {results_file_path}")

# Final best solution
np.savetxt(f"{experiment_name}/final_best.txt", global_best_position)
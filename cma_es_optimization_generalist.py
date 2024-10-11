

import numpy as np
import sys
import os
import cma
from evoman.environment import Environment
from player_controller import player_controller

experiment_name = 'cmaes_evoman'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = None

n_hidden = 10
n_inputs = 20  
controller = player_controller(n_hidden)
n_params = controller.get_n_params(n_inputs)

# Define the fitness function
def fitness_function(weights):
    total_gain = 0
    enemies = [1, 2, 3]  # List of enemies
    for enemy in enemies:
        # Create new environment for each enemy
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(n_hidden),
            enemymode="static",
            level=2,
            speed="fastest",
            multiplemode="no"
        )
        n_inputs = env.get_num_sensors()
        env.player_controller.set_weights(weights, n_inputs)
        f, p, e, t = env.play()
        gain = p - e
        total_gain += gain
    average_gain = total_gain / len(enemies)
    return -average_gain  # CMA-ES minimizes the function

# CMA-ES Optimization Loop
sigma = 0.5  # Initial standard deviation
population_size = 200  

# CMA-ES optimizer
es = cma.CMAEvolutionStrategy(n_params * [0], sigma, {'popsize': population_size})

n_generations = 50

for generation in range(n_generations):
    solutions = es.ask()
    fitnesses = []

    for solution in solutions:
        fitness = fitness_function(solution)
        fitnesses.append(fitness)

    es.tell(solutions, fitnesses)

    # Logging
    best_fitness = -np.min(fitnesses)
    print(f"Generation {generation+1}/{n_generations} - Best Fitness: {best_fitness}")

    if generation % 10 == 0:
        best_solution = es.best.get()[0]
        np.save(f"{experiment_name}/best_solution_gen_{generation}.npy", best_solution)


best_weights = es.best.get()[0]
np.save(f"{experiment_name}/best_solution.npy", best_weights)

# Test Best Controller
env = Environment(
    experiment_name=experiment_name,
    enemies=[1],  
    playermode="ai",
    player_controller=player_controller(n_hidden),
    enemymode="static",
    level=2,
    speed="normal"
)
n_inputs = env.get_num_sensors()
env.player_controller.set_weights(best_weights, n_inputs)
env.play()
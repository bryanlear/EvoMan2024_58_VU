from evoman.environment import Environment
from player_controller import player_controller
import numpy as np
import sys
import os

# Import CMA-ES optimizer
from cma import CMAEvolutionStrategy

# Parameters
n_generations = 30
population_size = 300
sigma = 0.5  # Initial standard deviation

# List of enemies to evaluate
enemies = [8]

# Loop over each enemy
for enemy in enemies:
    print(f"Starting evolution for Enemy {enemy}")

    experiment_name = f"enemy_{enemy}_experiment"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Create the environment for this enemy
    env = Environment(
        experiment_name=experiment_name,
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(10),  # n_hidden set to 10
        enemymode="static",
        level=2,
        speed="fastest",
        multiplemode="no"
    )

    n_inputs = env.get_num_sensors()
    n_hidden = 10  # Calculated to match 265 weights
    n_vars = n_hidden + (n_inputs * n_hidden) + (n_hidden * 5) + 5

    # Initialize CMA-ES
    x0 = np.random.randn(n_vars)
    es = CMAEvolutionStrategy(x0, sigma, {'popsize': population_size})

    # Evolutionary loop
    for generation in range(n_generations):
        solutions = es.ask()
        fitnesses = []

        for weights in solutions:
            # Set the weights to the controller
            env.player_controller.set(weights, n_inputs)

            # Run the simulation and obtain fitness
            f, p, e, t = env.play()
            gain = p - e
            fitness = -gain  # CMA-ES minimizes the fitness function
            fitnesses.append(fitness)

        es.tell(solutions, fitnesses)
        es.disp()
        best_fitness = min(fitnesses)
        print(f"Generation {generation+1}/{n_generations} - Best Fitness: {best_fitness}")

    # Save the best solution for this enemy
    best_solution = es.result.xbest
    np.save(f"best_weights_enemy_{enemy}.npy", best_solution)

    # Close the environment if necessary
    del env
    print(f"Completed evolution for Enemy {enemy}\n")
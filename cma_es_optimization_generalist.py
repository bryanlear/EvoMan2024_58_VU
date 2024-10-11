
import numpy as np
import sys
import os
import cma
from evoman.environment import Environment
from player_controller import player_controller
#  Experiment name
experiment_name = 'cmaes_evoman'

# Create directory
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


env = Environment(
    experiment_name='cmaes_evoman',
    enemies=[1, 2, 3], 
    playermode="ai",
    player_controller=player_controller(10),
    enemymode="static",
    level=2,
    speed="fastest"
)
# Number of inputs from environment
n_inputs = env.get_num_sensors()

# 1* Initialize controller 2* Computetotal number of parameters
controller = env.player_controller
n_params = controller.get_n_params(n_inputs)

# 2. Fitness Function
def fitness_function(weights):
    total_gain = 0
    enemies = [1, 2, 3]  
    for enemy in enemies:
        # Create a new environment for each enemy
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(10),
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
    return -average_gain  #negate

# 3. CMA-ES Optimization Loop
# CMA-ES parameters
sigma = 0.5  # Initial standard deviation
population_size = 100

# CMA-ES optimizer
es = cma.CMAEvolutionStrategy(n_params * [0], sigma, {'popsize': population_size})

# Number generations
n_generations = 30

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

    # Save best solution periodically
    if generation % 10 == 0:
        best_solution = es.best.get()[0]
        np.save(f"best_solution_gen_{generation}.npy", best_solution)

# 4. Retrieve and Save bestt Solution
best_weights = es.best.get()[0]
np.save("best_solution.npy", best_weights)

# 5. Test best controller
env.player_controller.set_weights(best_weights, n_inputs)
env.speed = "normal"
env.play()
from evoman.environment import Environment
from player_controller import player_controller
import numpy as np
import sys
import os 

# CMA-ES optimizer
from cma import CMAEvolutionStrategy

# Parameters
n_hidden = 25  # Number of hidden neurons in your controller
n_generations = 30
population_size = 200

# List of enemies to evaluate
enemies = [1, 2, 3]

# Loop over each enemy
for enemy in enemies:
    print(f"Starting evolution for Enemy {enemy}")
    

    experiment_name = f"enemy_{enemy}_experiment"
    
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # Create environment enemy
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
    n_vars = env.player_controller.get_n_params(n_inputs)
    
    # Initialize CMA-ES
    sigma = 0.5  # Initial standard deviation
    x0 = np.random.randn(n_vars)
    es = CMAEvolutionStrategy(x0, sigma, {'popsize': population_size})
    
    # Evolutionary loop
    for generation in range(n_generations):
        solutions = es.ask()
        fitnesses = []
        
        for weights in solutions:
            # Set weights controller
            env.player_controller.set_weights(weights, n_inputs)
            env.player_controller.reset() 
            
            # Run and obtain fitness
            f, p, e, t = env.play()
            gain = p - e
            fitness = -gain  # CMA-ES minimizes fitness function
            fitnesses.append(fitness)
        
        es.tell(solutions, fitnesses)
        es.disp()
        best_fitness = min(fitnesses)
        print(f"Generation {generation+1}/{n_generations} - Best Fitness: {best_fitness}")
    
    # Save best solution for enemy
    best_solution = es.result.xbest
    np.save(f"best_weights_enemy_{enemy}.npy", best_solution)
    
    # Close the environment if needed
    del env
    print(f"Completed evolution for Enemy {enemy}\n")
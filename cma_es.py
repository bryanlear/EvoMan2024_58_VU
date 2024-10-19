import sys
import os
import numpy as np
import cma  # Import the CMA-ES library
from evoman.environment import Environment
from demo_controller import player_controller

# Set random seed for reproducibility
np.random.seed(42)

# Experiment parameters
experiment_name = "CMAES_group2"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Create a run without the visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Parameters
best_individual_retries = 5
gens = 30  # Number of generations
runs = 10  # Number of runs

input_size = 20
hidden_neurons_size = 10
output_size = 5
total_genes = (input_size + 1) * hidden_neurons_size + (hidden_neurons_size + 1) * output_size

# Environment
env = Environment(experiment_name=experiment_name,
                  enemies=[1, 3, 5, 8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(hidden_neurons_size),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Simulation function
def simulation(env, individual_x):
    fitness, player_health, enemy_health, duration = env.play(pcont=individual_x)
    return fitness

# Optimization loop with adaptive sigma
for run in range(1, runs + 1):
    print(f"Starting run {run}/{runs}")

    # Initial solution (mean of the distribution)
    x0 = np.random.uniform(-1, 1, total_genes)
    sigma0 = 0.5  # Initial standard deviation

    # Set up CMA-ES options
    options = {
        'popsize': 200,
        'maxiter': gens,
        'bounds': [-1, 1],
        'verb_filenameprefix': os.path.join(experiment_name, f'log_run_{run}_'),
        'verb_log': 0,  # Disable CMA-ES logging to file
    }

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)

    # Prepare to record per-generation data
    results_file_path = os.path.join(experiment_name, f'results_CMAES_group1_run_{run}.txt')
    with open(results_file_path, 'w') as results_file:
        results_file.write('Generation,Best_fitness,Mean_fitness,SD_fitness,Sigma\n')

    # Variables for adaptive sigma
    best_fitness_overall = -np.inf
    no_improvement_counter = 0
    improvement_threshold = 5  # Number of generations to wait before adjusting sigma

    # Optimization loop
    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for individual in solutions:
            fitness = simulation(env, individual)
            fitnesses.append(fitness)

        es.tell(solutions, fitnesses)

        # Record per-generation data
        best_fitness = np.max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        sigma = es.sigma  # Get the current sigma value
        generation += 1

        # Adaptive sigma adjustment
        if best_fitness > best_fitness_overall:
            # Improvement found
            best_fitness_overall = best_fitness
            no_improvement_counter = 0
            # Decrease sigma to focus on exploitation
            es.sigma *= 0.9
            # Ensure sigma remains positive and within reasonable bounds
            es.sigma = max(es.sigma, 1e-5)
        else:
            # No improvement
            no_improvement_counter += 1
            if no_improvement_counter >= improvement_threshold:
                # Increase sigma to encourage exploration
                es.sigma *= 1.1
                no_improvement_counter = 0  # Reset counter after adjustment
                # Ensure sigma does not exceed initial sigma0
                es.sigma = min(es.sigma, sigma0)

        print(f"Run {run}, Generation {generation}/{gens}, Best Fitness: {best_fitness}, Sigma: {es.sigma}")

        with open(results_file_path, 'a') as results_file:
            results_file.write(f"{generation},{best_fitness},{mean_fitness},{std_fitness},{es.sigma}\n")

        # Save the best individual
        if generation == gens:
            best_individual = es.best.get()[0]
            best_individual_filename = os.path.join(experiment_name, f'best_weights_group1_run_{run}.npy')
            np.save(best_individual_filename, best_individual)
            break  # Exit after reaching the desired number of generations
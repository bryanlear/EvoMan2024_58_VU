import time
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up environment once
def setup_environment(experiment_name, enemy_num, n_hidden_neurons=10):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    env = Environment(
        experiment_name=experiment_name,
        enemies=[enemy_num],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )
    return env

# Runs the simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f, p, e

# Evaluates the population's fitness
def evaluate(env, population):
    return np.array([simulation(env, individual)[0] for individual in population])

# Main evolutionary function
def run_ga(env, mutation_rate, population_size, generations, doomsday_trigger, doomsday_percentage):
    n_vars = (env.get_num_sensors() + 1) * 10 + (10 + 1) * 5  # Number of weights
    population = np.random.uniform(-1, 1, (population_size, n_vars))

    stagnation_counter = 0
    best_fitness_last_gen = None

    # Create lists to store data for CSV file
    mean_fitness_list = []
    std_fitness_list = []
    max_fitness_list = []
    
    def crossover(parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(agent):
        mutation_mask = np.random.rand(len(agent)) < mutation_rate
        agent[mutation_mask] += np.random.normal(0, 0.1, mutation_mask.sum())
        return np.clip(agent, -1.0, 1.0)

    def select_parents(population, fitness):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        return population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]

    def apply_doomsday(population):
        num_doomsday = int(population_size * doomsday_percentage)
        population[-num_doomsday:] = np.random.uniform(-1, 1, (num_doomsday, n_vars))
        print(f"Doomsday triggered! Replacing {num_doomsday} individuals.")
        return population

    # Main evolutionary loop
    for generation in range(generations):
        fitness = evaluate(env, population)
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)
        max_fitness = np.max(fitness)
        
        # Store data for CSV file
        mean_fitness_list.append(mean_fitness)
        std_fitness_list.append(std_fitness)
        max_fitness_list.append(max_fitness)

        print(f"Generation {generation + 1}/{generations} - Mean Fitness: {mean_fitness}, Max Fitness: {max_fitness}, Std Dev: {std_fitness}")
        
        if best_fitness_last_gen is not None and max_fitness <= best_fitness_last_gen:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        best_fitness_last_gen = max_fitness

        # Trigger Doomsday if stagnation happens for doomsday_trigger generations
        if stagnation_counter >= doomsday_trigger:
            population = apply_doomsday(population)
            stagnation_counter = 0

        # Preserve elites and create new population
        elite_indices = np.argsort(fitness)[-2:]
        elites = population[elite_indices]
        new_population = list(elites)

        # Generate offspring with crossover and mutation
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness), select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = np.array(new_population)

    # Save mean, std dev, and max fitness values to a CSV file for this run
    results_df = pd.DataFrame({
        'Generation': np.arange(1, generations + 1),
        'MeanFitness': mean_fitness_list,
        'StdDevFitness': std_fitness_list,
        'MaxFitness': max_fitness_list
    })
    
    results_file_path = f"{env.experiment_name}/run_{time.time()}_results.csv"
    results_df.to_csv(results_file_path, index=False)
    print(f"Results saved to {results_file_path}")
    
    # Return the best individual from the population
    best_individual = population[np.argmax(fitness)]
    return best_individual

if __name__ == "__main__":
    # Define parameters and environment setup
    experiment_name = 'ga_test'
    enemy_num = 8
    num_runs = 10
    env = setup_environment(experiment_name, enemy_num)

    population_size = 100
    generations = 30
    mutation_rate = 0.1
    doomsday_trigger = 5
    doomsday_percentage = 0.2
    
    # Run GA for multiple runs
    best_solutions = []
    for run in range(num_runs):
        print(f"Starting run {run + 1}/{num_runs}")
        best_solution = run_ga(env, mutation_rate, population_size, generations, doomsday_trigger, doomsday_percentage)
        
        # Save the best solution for this run
        np.savetxt(f"{env.experiment_name}/final_best_run_{run + 1}.txt", best_solution)
        best_solutions.append(best_solution)

    print("GA optimization complete for all runs.")
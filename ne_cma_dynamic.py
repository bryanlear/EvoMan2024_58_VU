import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller
from cma import CMAEvolutionStrategy
from multiprocessing import Pool

# Parameters
n_hidden = 10
enemies = [1, 3, 5, 8]
ne_generations = 50  # Maximum generations for NE phase
cmaes_generations = 50  # Number of generations for CMA-ES phase
population_size = 100
stagnation_threshold = 10  # Number of generations with no significant improvement to trigger the switch

# Fitness function constants
alpha, beta, gamma = 1.0, 1.0, 0.1  # Weights for player health, enemy health, and time, respectively

# Track progress for plotting
fitness_log = []
best_fitness_ne = []
best_fitness_cmaes = []

# Fitness function
def evaluate_fitness(weights, enemy):
    env = Environment(experiment_name=f"enemy_{enemy}_hybrid",
                      enemies=[enemy], 
                      playermode="ai",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest")
    
    f, player_health, enemy_health, time_spent = env.play(pcont=weights)
    fitness = alpha * player_health - beta * enemy_health - gamma * time_spent
    return fitness

# Evaluation for the population (NE and CMA-ES will both use this)
def evaluate_population(population, enemy):
    with Pool(processes=4) as pool:
        fitnesses = pool.starmap(evaluate_fitness, [(ind, enemy) for ind in population])
    return fitnesses

# NE Phase: Evolve using simple mutation and crossover with dynamic transition
def ne_phase(enemy, population_size, n_vars):
    population = np.random.uniform(-1, 1, (population_size, n_vars))
    best_fitness = -np.inf
    stagnation_counter = 0
    
    for gen in range(ne_generations):
        # Evaluate fitness
        fitnesses = evaluate_population(population, enemy)
        
        # Track the best fitness
        gen_best_fitness = np.max(fitnesses)
        best_fitness_ne.append(gen_best_fitness)
        
        # Track progress for plotting
        fitness_log.append((gen, gen_best_fitness, 'NE'))
        
        # Check for stagnation
        if gen_best_fitness - best_fitness < 0.01:  # Small improvement
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_fitness = gen_best_fitness
        
        if stagnation_counter >= stagnation_threshold:
            print(f"Stagnation detected at generation {gen}. Switching to CMA-ES.")
            return population[np.argmax(fitnesses)]  # Return the best individual
        
        # Selection: Pick the top 50% individuals (simple truncation selection)
        sorted_indices = np.argsort(fitnesses)
        top_individuals = population[sorted_indices[:population_size // 2]]
        
        # Ensure we have an even number of top individuals for crossover
        if len(top_individuals) % 2 != 0:
            top_individuals = top_individuals[:-1]  # Remove the last individual if odd
        
        # Crossover: Pair and create offspring (single-point crossover)
        offspring = []
        for i in range(0, len(top_individuals), 2):
            parent1, parent2 = top_individuals[i], top_individuals[i + 1]
            cross_point = np.random.randint(0, len(parent1))
            child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
            child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
            offspring.extend([child1, child2])
        
        # Mutation: Randomly mutate offspring with a small probability
        mutation_rate = 0.1
        offspring = np.array(offspring)
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        offspring[mutation_mask] += np.random.randn(np.sum(mutation_mask))
        
        # Replace old population with offspring
        population = np.vstack([top_individuals, offspring])
    
    # Return the best individual if NE phase completes
    return population[np.argmax(fitnesses)]

# CMA-ES Phase: Fine-tune the best individual from NE phase
def cmaes_phase(best_individual, enemy, n_generations, sigma=0.5):
    es = CMAEvolutionStrategy(best_individual, sigma, {'popsize': population_size})
    
    for gen in range(n_generations):
        solutions = es.ask()
        fitnesses = evaluate_population(solutions, enemy)
        
        # Track the best fitness for plotting
        gen_best_fitness = np.max(fitnesses)
        best_fitness_cmaes.append(gen_best_fitness)
        fitness_log.append((gen + ne_generations, gen_best_fitness, 'CMA-ES'))
        
        es.tell(solutions, fitnesses)
        es.disp()
    
    # Return the best solution after CMA-ES
    return es.result.xbest

# Main loop to run the hybrid algorithm
if __name__ == "__main__":
    for enemy in enemies:
        # Set up environment
        env = Environment(experiment_name=f"enemy_{enemy}_hybrid",
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(n_hidden),
                          enemymode="static",
                          level=2,
                          speed="fastest")

        # Calculate n_vars (number of weights in the neural network)
        n_inputs = env.get_num_sensors()  # Get the number of inputs from the environment
        n_vars = (n_inputs + 1) * n_hidden + (n_hidden + 1) * 5  # Formula to calculate total number of weights
        
        # Phase 1: NE Exploration with dynamic transition
        print(f"Starting NE phase for Enemy {enemy}")
        best_individual_ne = ne_phase(enemy, population_size, n_vars)
        
        # Phase 2: CMA-ES Exploitation
        print(f"Switching to CMA-ES phase for Enemy {enemy}")
        best_individual_cmaes = cmaes_phase(best_individual_ne, enemy, cmaes_generations)
        
        # Save the final best individual
        np.save(f"best_weights_enemy_{enemy}.npy", best_individual_cmaes)
        
        # Plot fitness progress
        generations, fitness_values, phases = zip(*fitness_log)
        plt.plot(generations, fitness_values, label="Fitness")
        plt.axvline(x=ne_generations, color='r', linestyle='--', label="CMA-ES Start")
        plt.title(f"Fitness over Generations for Enemy {enemy}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig(f"fitness_progress_enemy_{enemy}.png")
        plt.close()
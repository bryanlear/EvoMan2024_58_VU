import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_generations = 30  
num_runs = 10  
groups = ['Group1', 'Group2']

# Function to load fitness data
def load_fitness_data(algorithm, group_name, run):
    file_path = f"{algorithm}_{group_name}_run_{run}/fitness_history_{group_name}_run_{run}.npy"
    return np.load(file_path)

# Compute the best and mean fitness across runs
def get_best_mean_fitness(algorithm, group_name):
    best_fitness = []
    mean_fitness = []
    
    for generation in range(num_generations):
        fitness_for_generation = []
        
        for run in range(1, num_runs + 1):
            fitness_history = load_fitness_data(algorithm, group_name, run)
            fitness_for_generation.append(fitness_history[generation])
        
        # Compute best and mean fitness for current generation
        best_fitness.append(np.min(fitness_for_generation))
        mean_fitness.append(np.mean(fitness_for_generation))
    
    return best_fitness, mean_fitness

# Plot fitness over generations
def plot_fitness(algorithm1, algorithm2, group_name):
    # Load fitness data for both algorithms
    best_fitness_alg1, mean_fitness_alg1 = get_best_mean_fitness(algorithm1, group_name)
    best_fitness_alg2, mean_fitness_alg2 = get_best_mean_fitness(algorithm2, group_name)

    # Plot best and mean fitness for both algorithms
    plt.figure()
    plt.plot(best_fitness_alg1, label=f"Best {algorithm1}", color='blue')
    plt.plot(mean_fitness_alg1, label=f"Mean {algorithm1}", linestyle='--', color='blue')
    plt.plot(best_fitness_alg2, label=f"Best {algorithm2}", color='red')
    plt.plot(mean_fitness_alg2, label=f"Mean {algorithm2}", linestyle='--', color='red')

    # Add titles and labels
    plt.title(f'Fitness Over Generations ({group_name})')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'fitness_plot_{group_name}.png')
    plt.show()

# Plot fitness for both groups
plot_fitness('CMA-ES', 'NE', 'Group1')
plot_fitness('CMA-ES', 'NE', 'Group2')
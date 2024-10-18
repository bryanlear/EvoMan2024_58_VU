import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
num_generations = 30
num_runs = 10
groups = ['Group1', 'Group2']

# Base directories where your experiment data is stored
base_dir_cma_es = "/Users/bry_lee/EvoMan2024_58_VU/CMA-ES"
base_dir_ne = "/Users/bry_lee/EvoMan2024_58_VU/NE/results_tests"  # Adjust if necessary

# Function to load fitness data
def load_fitness_data(algorithm, group_name, run):
    if algorithm == 'CMA-ES':
        file_path = os.path.join(base_dir_cma_es, f"{group_name}_run_{run}", f"scaled_fitness_history_{group_name}_run_{run}.npy")
    elif algorithm == 'NE':
        file_path = os.path.join(base_dir_ne, f"{group_name}_final", f"NE_group1_best_individual_{run}.txt")  
    else:
        print(f"Unknown algorithm: {algorithm}")
        return None
    
    # Check if the file exists before loading
    if os.path.exists(file_path):
        return np.load(file_path) if algorithm == 'CMA-ES' else np.loadtxt(file_path)
    else:
        print(f"Fitness history file not found: {file_path}")
        return None

# Function to compute the best and mean fitness across runs
def get_best_mean_fitness(algorithm, group_name):
    best_fitness = []
    mean_fitness = []
    all_fitnesses = []
    
    for generation in range(num_generations):
        fitness_for_generation = []
        
        for run in range(1, num_runs + 1):
            fitness_history = load_fitness_data(algorithm, group_name, run)
            
            if fitness_history is not None:
                fitness_for_generation.append(fitness_history[generation] if algorithm == 'CMA-ES' else fitness_history[run])
        
        if fitness_for_generation:
            best_fitness.append(np.min(fitness_for_generation))
            mean_fitness.append(np.mean(fitness_for_generation))
            all_fitnesses.append(fitness_for_generation)
    
    return best_fitness, mean_fitness, all_fitnesses

# Function to plot fitness over generations
def plot_fitness(algorithm1, algorithm2, group_name):
    # Load fitness data for both algorithms
    best_fitness_alg1, mean_fitness_alg1, all_fitness_alg1 = get_best_mean_fitness(algorithm1, group_name)
    best_fitness_alg2, mean_fitness_alg2, all_fitness_alg2 = get_best_mean_fitness(algorithm2, group_name)

    # Compute standard deviations for the shaded areas
    std_fitness_alg1 = np.std(all_fitness_alg1, axis=1)
    std_fitness_alg2 = np.std(all_fitness_alg2, axis=1)

    # Plot best and mean fitness with shaded regions for both algorithms
    plt.figure()

    # CMA-ES plotting
    plt.plot(best_fitness_alg1, label=f"Best {algorithm1} {group_name}", color='green')
    plt.plot(mean_fitness_alg1, label=f"Mean {algorithm1} {group_name}", linestyle='--', color='green')
    plt.fill_between(range(num_generations), 
                     np.array(mean_fitness_alg1) - std_fitness_alg1, 
                     np.array(mean_fitness_alg1) + std_fitness_alg1, 
                     color='green', alpha=0.2)

    # NE plotting
    plt.plot(best_fitness_alg2, label=f"Best {algorithm2} {group_name}", color='blue')
    plt.plot(mean_fitness_alg2, label=f"Mean {algorithm2} {group_name}", linestyle='--', color='blue')
    plt.fill_between(range(num_generations), 
                     np.array(mean_fitness_alg2) - std_fitness_alg2, 
                     np.array(mean_fitness_alg2) + std_fitness_alg2, 
                     color='blue', alpha=0.2)

    # Add titles and labels
    plt.title(f'Group {group_name} - Line Plot of Best and Mean Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(0, 100)  # Setting y-axis limit for fitness

    # Save and display
    plt.savefig(f'fitness_plot_{group_name}.png')
    plt.show()

# Plot fitness for both groups
plot_fitness('CMA-ES', 'NE', 'Group1')
plot_fitness('CMA-ES', 'NE', 'Group2')
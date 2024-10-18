import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Parameters
num_generations = 30
num_runs = 10
groups = ['Group1', 'Group2']

# Base directories where your experiment data is stored
base_dir_cma_es = "/CMA-ES"
base_dir_ne = "/NE/results_tests"

# Function to load CMA-ES fitness data
def load_cmaes_fitness_data(group_name, run):
    file_path = os.path.join(base_dir_cma_es, f"{group_name}_run_{run}", f"scaled_fitness_history_{group_name}_run_{run}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"Fitness history file not found: {file_path}")
        return None

# Function to load NE fitness data using pandas
def load_ne_fitness_data(group_name):
    file_path = os.path.join(base_dir_ne, f"{group_name}_final", f"NE_group1_results_.txt")  # Adjust path as needed
    if os.path.exists(file_path):
        # Manually specify column names based on the data structure we saw
        column_names = ['Run', 'Generation', 'Best_fitness', 'Mean_fitness', 'SD_mean_fitness']
        data = pd.read_csv(file_path, sep=',', header=None, names=column_names)
        return data
    else:
        print(f"NE fitness file not found: {file_path}")
        return None

# Function to compute the best and mean fitness for NE using pandas
def get_ne_best_mean_fitness(group_name):
    data = load_ne_fitness_data(group_name)
    if data is not None:
        # Group by 'Generation' and compute mean and standard deviation
        best_fit_mean = data.groupby('Generation')['Best_fitness'].mean()
        best_fit_sd = data.groupby('Generation')['Best_fitness'].std()

        mean_fit_mean = data.groupby('Generation')['Mean_fitness'].mean()
        mean_fit_sd = data.groupby('Generation')['Mean_fitness'].std()

        return best_fit_mean.values, mean_fit_mean.values, best_fit_sd.values, mean_fit_sd.values
    return None, None, None, None


def get_cmaes_best_mean_fitness(group_name):
    best_fitness = []
    mean_fitness = []
    all_fitnesses = []
    
    for generation in range(num_generations):
        fitness_for_generation = []
        
        for run in range(1, num_runs + 1):
            fitness_history = load_cmaes_fitness_data(group_name, run)
            
            if fitness_history is not None:
                fitness_for_generation.append(fitness_history[generation])
        
        if fitness_for_generation:
        
            best_fitness_for_gen = np.mean(fitness_for_generation)  
            mean_fitness_for_gen = np.min(fitness_for_generation)   

            best_fitness.append(best_fitness_for_gen)
            mean_fitness.append(mean_fitness_for_gen)
            all_fitnesses.append(fitness_for_generation)
    
    return best_fitness, mean_fitness, all_fitnesses

# Moving average function for smoothing curves
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Function to plot fitness over generations
def plot_fitness(group_name):
    # Load CMA-ES fitness data
    best_fitness_cmaes, mean_fitness_cmaes, all_fitness_cmaes = get_cmaes_best_mean_fitness(group_name)
    std_fitness_cmaes = np.std(all_fitness_cmaes, axis=1)

    # Smooth fitness curves with a moving average
    best_fitness_cmaes_smoothed = moving_average(best_fitness_cmaes, window_size=3)
    mean_fitness_cmaes_smoothed = moving_average(mean_fitness_cmaes, window_size=3)

    # Load NE fitness data
    best_fitness_ne, mean_fitness_ne, std_best_ne, std_mean_ne = get_ne_best_mean_fitness(group_name)

    # Plot best and mean fitness with shaded regions for both algorithms
    plt.figure()

    # CMA-ES plotting (smoothed values)
    plt.plot(range(len(best_fitness_cmaes_smoothed)), best_fitness_cmaes_smoothed, label=f"Best CMA-ES {group_name}", color='green')
    plt.plot(range(len(mean_fitness_cmaes_smoothed)), mean_fitness_cmaes_smoothed, label=f"Mean CMA-ES {group_name}", linestyle='--', color='green')
    plt.fill_between(range(len(mean_fitness_cmaes_smoothed)), 
                     np.array(mean_fitness_cmaes_smoothed) - std_fitness_cmaes[1:-1], 
                     np.array(mean_fitness_cmaes_smoothed) + std_fitness_cmaes[1:-1], 
                     color='green', alpha=0.2)

    # NE plotting
    if best_fitness_ne is not None:
        plt.plot(best_fitness_ne, label=f"Best NE {group_name}", color='blue')
        plt.plot(mean_fitness_ne, label=f"Mean NE {group_name}", linestyle='--', color='blue')
        plt.fill_between(range(num_generations), 
                         np.array(mean_fitness_ne) - std_mean_ne, 
                         np.array(mean_fitness_ne) + std_mean_ne, 
                         color='blue', alpha=0.2)

    # Add titles and labels
    plt.title(f'Group {group_name} - Line Plot of Best and Mean Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend(loc='upper left')
    plt.ylim(0, 100)

    # Save and display
    plt.savefig(f'fitness_plot_{group_name}.png')
    plt.show()

# Plot fitness for both groups
plot_fitness('Group1')
plot_fitness('Group2')
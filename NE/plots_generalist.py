# import numpy as np
# import matplotlib.pyplot as plt

# # Set the enemy folder dynamically
# enemy = 'enemy1'  # Change this to 'enemy1', 'enemy2', or 'enemy3' as needed

# # Base path
# base_path = f'/Users/hanacosip/Desktop/MASTER/Evolutionary Computing/Task2EC/Task2EC/results_tests/{enemy}/'

# # File paths for the 10 runs
# file_paths = [
#     base_path + 'NE_group1_best_individual_1.txt',
#     base_path + 'NE_group1_best_individual_2.txt',
#     base_path + 'NE_group1_best_individual_3.txt',
#     base_path + 'NE_group1_best_individual_4.txt',
#     base_path + 'NE_group1_best_individual_5.txt',
#     base_path + 'NE_group1_best_individual_6.txt',
#     base_path + 'NE_group1_best_individual_7.txt',
#     base_path + 'NE_group1_best_individual_8.txt',
#     base_path + 'NE_group1_best_individual_9.txt',
#     base_path + 'NE_group1_best_individual_10.txt'
# ]

# # Initialize lists to store the data for each generation
# mean_fitness_per_run = []
# max_fitness_per_run = []

# # Read each file and extract the mean and max fitness
# for file_path in file_paths:
#     data = np.loadtxt(file_path, delimiter=',')  # Assuming the file is comma-delimited
#     generations = data[:, 0]  # First column: generations
#     fitness_scores = data[:, 2]  # Third column: fitness scores
    
#     mean_fitness_per_run.append(fitness_scores.mean())  # Calculate mean fitness
#     max_fitness_per_run.append(fitness_scores.max())    # Calculate max fitness

# # Convert lists to arrays for easier computation
# mean_fitness_per_run = np.array(mean_fitness_per_run)
# max_fitness_per_run = np.array(max_fitness_per_run)

# # Calculate mean and standard deviation across runs
# mean_of_means = np.mean(mean_fitness_per_run, axis=0)
# std_of_means = np.std(mean_fitness_per_run, axis=0)

# mean_of_maxs = np.mean(max_fitness_per_run, axis=0)
# std_of_maxs = np.std(max_fitness_per_run, axis=0)

# # Plotting
# plt.figure(figsize=(10, 6))

# # Plot mean fitness
# plt.errorbar(generations, mean_of_means, yerr=std_of_means, label='Mean Fitness', fmt='-o')

# # Plot max fitness
# plt.errorbar(generations, mean_of_maxs, yerr=std_of_maxs, label='Max Fitness', fmt='-o')

# plt.title(f'Mean and Max Fitness Across Generations ({enemy})')
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set the enemy folder dynamically
enemy = 'enemy1'  # Change this to 'enemy1', 'enemy2', or 'enemy3' as needed

# Parameters
gens = 30  # Number of generations
runs = 10  # Number of runs

# Base path
base_path = f'/Users/hanacosip/Desktop/MASTER/Evolutionary Computing/Task2EC/Task2EC/results_tests/{enemy}/'

# File paths for the 10 runs
file_paths = [
    base_path + f'NE_group1_best_individual_{i}.txt' for i in range(1, runs + 1)
]

# Initialize lists to store the data for each generation
all_runs_mean_fitness = []
all_runs_best_fitness = []

# Handle different formats based on the number of columns
for file_path in file_paths:
    try:
        data = np.loadtxt(file_path, delimiter=',')
        
        # If the file has a single column, treat it as fitness data for consecutive generations
        if data.ndim == 1:
            generations = np.arange(1, gens + 1)  # Assume each line corresponds to a generation
            fitness_scores = data[:gens]  # Ensure we handle only 30 generations
            
        # If the file has three columns, use the third column for fitness data
        elif data.shape[1] == 3:
            generations = data[:, 0]  # First column: generations
            fitness_scores = data[:, 2]  # Third column: fitness scores
        
        # Collect mean fitness and best fitness for the current run
        all_runs_mean_fitness.append(fitness_scores)
        all_runs_best_fitness.append(fitness_scores.max())
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Convert lists to arrays for easier computation
all_runs_mean_fitness = np.array(all_runs_mean_fitness)

# Calculate mean and standard deviation across runs for each generation
mean_fitness = np.mean(all_runs_mean_fitness, axis=0)
std_fitness = np.std(all_runs_mean_fitness, axis=0)

# Calculate the best fitness across all runs
best_fitness = np.max(all_runs_mean_fitness, axis=0)

# Plotting
plt.figure(figsize=(10, 6))

# Plot mean fitness with error bars
plt.errorbar(np.arange(1, gens + 1), mean_fitness, yerr=std_fitness, label='Mean Fitness', fmt='-o', color='blue')

# Plot best fitness
plt.plot(np.arange(1, gens + 1), best_fitness, label='Best Fitness', '-', color='red')

plt.title(f'Mean and Best Fitness Across Generations ({enemy})')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

gens = 30  # Number of generations
runs = 10  # Number of runs
groups = ['CMAES_group1', 'CMAES_group2']  # Adjust based on available data

all_data = {}
for group in groups:
    all_data[group] = {'best_fitness': [], 'mean_fitness': []}

# Set base_path to the directory of this script
base_path = os.path.dirname(os.path.abspath(__file__))
print(f"Base path is: {base_path}")

generations = np.arange(1, gens + 1)

for group in groups:
    for run in range(1, runs + 1):
        results_file = os.path.join(base_path, group, f'results_{group}_run_{run}.txt')
        print(f"Trying to read {results_file}")
        try:
            data = pd.read_csv(results_file, header=0)
            # Ensure the data has the correct number of generations
            data = data.iloc[:gens]
            # Store the fitness values
            all_data[group]['best_fitness'].append(data['Best_fitness'].values)
            all_data[group]['mean_fitness'].append(data['Mean_fitness'].values)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")

    # Convert lists to numpy arrays
    best_fitness_array = np.array(all_data[group]['best_fitness'])
    mean_fitness_array = np.array(all_data[group]['mean_fitness'])

    # Check if arrays are not empty
    if best_fitness_array.size == 0 or mean_fitness_array.size == 0:
        print(f"No data found for group {group}. Skipping plotting.")
        continue

    # Compute mean and std across runs for each generation
    best_fitness_mean = np.mean(best_fitness_array, axis=0)
    best_fitness_std = np.std(best_fitness_array, axis=0)

    mean_fitness_mean = np.mean(mean_fitness_array, axis=0)
    mean_fitness_std = np.std(mean_fitness_array, axis=0)

    # Store back in the dictionary
    all_data[group]['best_fitness_mean'] = best_fitness_mean
    all_data[group]['best_fitness_std'] = best_fitness_std
    all_data[group]['mean_fitness_mean'] = mean_fitness_mean
    all_data[group]['mean_fitness_std'] = mean_fitness_std

    # Create a figure for each group
    plt.figure(figsize=(10, 6))

    # Plot Best Fitness with shaded std deviation
    plt.plot(generations, all_data[group]['best_fitness_mean'], label=f'Best Fitness {group}', color='blue')
    plt.fill_between(generations, 
                     all_data[group]['best_fitness_mean'] - all_data[group]['best_fitness_std'],
                     all_data[group]['best_fitness_mean'] + all_data[group]['best_fitness_std'],
                     color='blue', alpha=0.3)

    # Plot Mean Fitness with shaded std deviation
    plt.plot(generations, all_data[group]['mean_fitness_mean'], label=f'Mean Fitness {group}', color='green')
    plt.fill_between(generations, 
                     all_data[group]['mean_fitness_mean'] - all_data[group]['mean_fitness_std'],
                     all_data[group]['mean_fitness_mean'] + all_data[group]['mean_fitness_std'],
                     color='green', alpha=0.3)

    # Set titles and labels as per the sample charts
    plt.title(f'Group {group} line plot of best and mean performance')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')

    # Add legend and remove grid
    plt.legend()
    
    # Remove the grid (comment out plt.grid)
    # plt.grid(True)  # Removed to match your chart style

    # Save the plot to file
    plt.savefig(os.path.join(base_path, group, f'fitness_plot_{group}.png'))
    plt.show()

# Commented out undefined functions
# for group in groups:
#     for run in range(1, runs + 1):
#         best_individual_filename = os.path.join(base_path, group, f'best_weights_{group}_run_{run}.npy')
#         best_individual = np.load(best_individual_filename)

#         # Evaluate the best individual
#         fitness = simulation(env, best_individual)
#         print(f"Group {group}, Run {run}, Best Individual Fitness: {fitness}")
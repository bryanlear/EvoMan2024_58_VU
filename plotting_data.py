import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_generations = 30
num_runs = 10
groups = ['group1', 'group2']

def read_cmaes_data(group):
    cmaes_data = []
    for run in range(1, num_runs + 1):
        # Construct file path
        file_path = os.path.join(f'CMAES_{group}', f'results_CMAES_{group}_run_{run}.txt')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            cmaes_data.append(data)
        else:
            print(f'File not found: {file_path}')
    return cmaes_data

def read_ne_data(group):
    # Handle inconsistency in file naming for Group2
    ne_file = os.path.join('NE', 'results_tests', f'Group{group[-1]}_final', f'NE_group1_results_.txt')
    if os.path.exists(ne_file):
        data = pd.read_csv(ne_file, header=None)
        # Assuming data has columns: ['Run', 'Generation', 'Best_fitness', 'Mean_fitness', 'SD_mean_fitness']
        data.columns = ['Run', 'Generation', 'Best_fitness', 'Mean_fitness', 'SD_mean_fitness']
        return data
    else:
        print(f'File not found: {ne_file}')
        return None

def process_cmaes_data(cmaes_data):
    best_fitness_runs = []
    mean_fitness_runs = []
    for df in cmaes_data:
        df = df.iloc[:num_generations]  # Ensure we have num_generations rows
        best_fitness_runs.append(df['Best_fitness'].values)
        mean_fitness_runs.append(df['Mean_fitness'].values)
    # Convert to numpy arrays
    best_fitness_runs = np.array(best_fitness_runs)  # Shape: (num_runs, num_generations)
    mean_fitness_runs = np.array(mean_fitness_runs)
    # Compute mean and std over runs
    best_fitness_mean = np.mean(best_fitness_runs, axis=0)
    best_fitness_std = np.std(best_fitness_runs, axis=0)
    mean_fitness_mean = np.mean(mean_fitness_runs, axis=0)
    mean_fitness_std = np.std(mean_fitness_runs, axis=0)
    return best_fitness_mean, best_fitness_std, mean_fitness_mean, mean_fitness_std

def process_ne_data(ne_data):
    # ne_data is a DataFrame containing data from all runs
    # We need to group by Generation and compute mean and std
    grouped = ne_data.groupby('Generation')
    best_fitness_mean = grouped['Best_fitness'].mean().values[:num_generations]
    best_fitness_std = grouped['Best_fitness'].std().values[:num_generations]
    mean_fitness_mean = grouped['Mean_fitness'].mean().values[:num_generations]
    mean_fitness_std = grouped['Mean_fitness'].std().values[:num_generations]
    return best_fitness_mean, best_fitness_std, mean_fitness_mean, mean_fitness_std

def plot_fitness(group, cmaes_results, ne_results):
    generations = np.arange(1, num_generations + 1)
    # Unpack CMA-ES results
    cmaes_best_mean, cmaes_best_std, cmaes_mean_mean, cmaes_mean_std = cmaes_results
    # Unpack NE results
    ne_best_mean, ne_best_std, ne_mean_mean, ne_mean_std = ne_results
    
    plt.figure(figsize=(10,6))

    # Colors from the screenshot for each algorithm
    cmaes_color = '#2ca02c'  # Green
    ne_color = '#1f77b4'     # Blue
    ne_mean_color = '#ff7f0e'  # Orange
    cmaes_mean_color = '#d62728'  # Red

    # Plot CMA-ES best fitness with shaded standard deviation
    plt.plot(generations, cmaes_best_mean, label='Best CMA-ES', color=cmaes_color)
    plt.fill_between(generations, 
                     cmaes_best_mean - cmaes_best_std, 
                     cmaes_best_mean + cmaes_best_std, 
                     color=cmaes_color, alpha=0.3)

    # Plot NE best fitness with shaded standard deviation
    plt.plot(generations, ne_best_mean, label='Best NE', color=ne_color)
    plt.fill_between(generations, 
                     ne_best_mean - ne_best_std, 
                     ne_best_mean + ne_best_std, 
                     color=ne_color, alpha=0.3)

    # Plot CMA-ES mean fitness with shaded standard deviation
    plt.plot(generations, cmaes_mean_mean, label='Mean CMA-ES', color=cmaes_mean_color, linestyle='--')
    plt.fill_between(generations, 
                     cmaes_mean_mean - cmaes_mean_std, 
                     cmaes_mean_mean + cmaes_mean_std, 
                     color=cmaes_mean_color, alpha=0.3)

    # Plot NE mean fitness with shaded standard deviation
    plt.plot(generations, ne_mean_mean, label='Mean NE', color=ne_mean_color, linestyle='--')
    plt.fill_between(generations, 
                     ne_mean_mean - ne_mean_std, 
                     ne_mean_mean + ne_mean_std, 
                     color=ne_mean_color, alpha=0.3)

    # Set titles and labels to match the format in your screenshot
    plt.title(f'Group {group.capitalize()} line plot of best and mean performance')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend(loc='upper right')

    # Save and show the plot
    plt.savefig(f'fitness_plot_{group}.png')
    plt.show()

# Now, for each group, read the data and plot
for group in groups:
    print(f"\nProcessing data for {group.capitalize()}...")
    # Read CMA-ES data
    cmaes_data = read_cmaes_data(group)
    if cmaes_data:
        cmaes_results = process_cmaes_data(cmaes_data)
    else:
        print(f'No CMA-ES data for {group}')
        continue  # Skip to the next group
    # Read NE data
    ne_data = read_ne_data(group)
    if ne_data is not None:
        ne_results = process_ne_data(ne_data)
    else:
        print(f'No NE data for {group}')
        continue  # Skip to the next group
    # Plot the results
    plot_fitness(group, cmaes_results, ne_results)
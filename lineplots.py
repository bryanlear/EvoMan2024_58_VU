import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define # enemies and runs
enemies = [2, 5, 8]
n_runs = 10

# Iterate over each enemy
for enemy in enemies:
    mean_values_all_runs = []
    max_values_all_runs = []

    # Load data for all 10 runs
    for run in range(1, n_runs + 1):
        file_path = f'evoman_experiments/enemy_{enemy}/{run}/run_*_results.csv' 

        # Ensure we select correct file
        matching_files = [f for f in os.listdir(f'evoman_experiments/enemy_{enemy}/{run}') if f.endswith('_results.csv')]
        if len(matching_files) != 1:
            print(f"Warning: Unexpected number of CSV files found in 'enemy_{enemy}/run_{run}'")
            continue
        
        # Load CSV
        file_path = f'evoman_experiments/enemy_{enemy}/{run}/{matching_files[0]}'
        df = pd.read_csv(file_path)
        
        # Collect mean, max values for each generation
        mean_values_all_runs.append(df['MeanFitness'].values)
        max_values_all_runs.append(df['MaxFitness'].values)

    # Convert to numpy arrays for easier calculations
    mean_values_all_runs = np.array(mean_values_all_runs)
    max_values_all_runs = np.array(max_values_all_runs)

    # Calculate average, standard deviation for each generation
    mean_avg = np.mean(mean_values_all_runs, axis=0)
    mean_std = np.std(mean_values_all_runs, axis=0)
    max_avg = np.mean(max_values_all_runs, axis=0)
    max_std = np.std(max_values_all_runs, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    generations = np.arange(1, len(mean_avg) + 1)

    plt.plot(generations, mean_avg, label='Mean Fitness', color='blue')
    plt.fill_between(generations, mean_avg - mean_std, mean_avg + mean_std, alpha=0.2, color='blue')

    plt.plot(generations, max_avg, label='Max Fitness', color='orange')
    plt.fill_between(generations, max_avg - max_std, max_avg + max_std, alpha=0.2, color='orange')

    plt.title(f'Enemy {enemy} - PSO Fitness Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = f'evoman_experiments/enemy_{enemy}/fitness_plot_enemy_{enemy}.png'
    plt.savefig(save_path)
    print(f'Saved plot for Enemy {enemy} at: {save_path}')
    plt.show()
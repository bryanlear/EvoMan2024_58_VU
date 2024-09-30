import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define enemies and number of runs
enemies = [2, 5, 8]
n_runs = 10

# Iterate over each enemy
for enemy in enemies:
    # Storage for PSO data
    pso_mean_values_all_runs = []
    pso_max_values_all_runs = []

    # Storage for GA data
    ga_mean_values_all_runs = []
    ga_max_values_all_runs = []

    # Load PSO data for all 10 runs
    for run in range(1, n_runs + 1):
        # Load PSO data
        pso_file_path = f'evoman_experiments/enemy_{enemy}/{run}/run_*_results.csv'

        # Ensure we select the correct file
        pso_matching_files = [f for f in os.listdir(f'evoman_experiments/enemy_{enemy}/{run}') if f.endswith('_results.csv')]
        if len(pso_matching_files) != 1:
            print(f"Warning: Unexpected number of CSV files found in 'enemy_{enemy}/run_{run}' for PSO")
            continue
        
        # Load PSO CSV
        pso_file_path = f'evoman_experiments/enemy_{enemy}/{run}/{pso_matching_files[0]}'
        pso_df = pd.read_csv(pso_file_path)
        
        # Collect mean, max values for PSO for each generation
        pso_mean_values_all_runs.append(pso_df['MeanFitness'].values)
        pso_max_values_all_runs.append(pso_df['MaxFitness'].values)

        # Load GA data
        ga_file_path = f'evoman_experiments_ga/enemy_{enemy}/{run}/run_*_results.csv'

        # Ensure we select the correct file
        ga_matching_files = [f for f in os.listdir(f'evoman_experiments_ga/enemy_{enemy}/{run}') if f.endswith('_results.csv')]
        if len(ga_matching_files) != 1:
            print(f"Warning: Unexpected number of CSV files found in 'enemy_{enemy}/run_{run}' for GA")
            continue
        
        # Load GA CSV
        ga_file_path = f'evoman_experiments_ga/enemy_{enemy}/{run}/{ga_matching_files[0]}'
        ga_df = pd.read_csv(ga_file_path)
        
        # Collect mean, max values for GA for each generation
        ga_mean_values_all_runs.append(ga_df['MeanFitness'].values)
        ga_max_values_all_runs.append(ga_df['MaxFitness'].values)

    # Convert to numpy arrays for easier calculations
    pso_mean_values_all_runs = np.array(pso_mean_values_all_runs)
    pso_max_values_all_runs = np.array(pso_max_values_all_runs)
    ga_mean_values_all_runs = np.array(ga_mean_values_all_runs)
    ga_max_values_all_runs = np.array(ga_max_values_all_runs)

    # Calculate average and standard deviation for each generation for PSO
    pso_mean_avg = np.mean(pso_mean_values_all_runs, axis=0)
    pso_mean_std = np.std(pso_mean_values_all_runs, axis=0)
    pso_max_avg = np.mean(pso_max_values_all_runs, axis=0)
    pso_max_std = np.std(pso_max_values_all_runs, axis=0)

    # Calculate average and standard deviation for each generation for GA
    ga_mean_avg = np.mean(ga_mean_values_all_runs, axis=0)
    ga_mean_std = np.std(ga_mean_values_all_runs, axis=0)
    ga_max_avg = np.mean(ga_max_values_all_runs, axis=0)
    ga_max_std = np.std(ga_max_values_all_runs, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    generations = np.arange(1, len(pso_mean_avg) + 1)

    # PSO plots
    plt.plot(generations, pso_mean_avg, label='PSO Mean Fitness', color='blue')
    plt.fill_between(generations, pso_mean_avg - pso_mean_std, pso_mean_avg + pso_mean_std, alpha=0.2, color='blue')

    plt.plot(generations, pso_max_avg, label='PSO Max Fitness', color='orange')
    plt.fill_between(generations, pso_max_avg - pso_max_std, pso_max_avg + pso_max_std, alpha=0.2, color='orange')

    # GA plots
    plt.plot(generations, ga_mean_avg, label='GA Mean Fitness', color='green')
    plt.fill_between(generations, ga_mean_avg - ga_mean_std, ga_mean_avg + ga_mean_std, alpha=0.2, color='green')

    plt.plot(generations, ga_max_avg, label='GA Max Fitness', color='red')
    plt.fill_between(generations, ga_max_avg - ga_max_std, ga_max_avg + ga_max_std, alpha=0.2, color='red')

    plt.title(f'Enemy {enemy} - PSO vs GA Fitness Across Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = f'evoman_experiments_ga/fitness_comparison_plot_enemy_{enemy}.png'
    plt.savefig(save_path)
    print(f'Saved plot for Enemy {enemy} at: {save_path}')
    plt.show()
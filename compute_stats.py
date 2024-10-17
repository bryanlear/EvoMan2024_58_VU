import numpy as np
import matplotlib.pyplot as plt
import os

enemies = [1, 2, 3, 4, 5, 6, 7, 8]
num_runs = 10  # Set this based on your structure, or automate it

def aggregate_fitness(enemy):
    all_fitness_histories = []
    max_length = 0  # To track the maximum length of fitness histories
    
    for run in range(1, num_runs + 1):
        experiment_name = f"enemy_{enemy}_run_{run}"
        file_path = f"{experiment_name}/fitness_history_enemy_{enemy}_run_{run}.npy"
        
        # Check if the file exists and load it if present
        if os.path.exists(file_path):
            try:
                fitness_history = np.load(file_path)
                all_fitness_histories.append(fitness_history)
                if len(fitness_history) > max_length:
                    max_length = len(fitness_history)  # Update the maximum length
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    if all_fitness_histories:
        # Pad each array with NaN values to make them all the same length
        padded_histories = []
        for fh in all_fitness_histories:
            padded = np.pad(fh, (0, max_length - len(fh)), 'constant', constant_values=np.nan)
            padded_histories.append(padded)

        # Stack all padded histories for statistical analysis
        all_fitness_histories = np.array(padded_histories)

        # Compute mean and std deviation across runs, ignoring NaN values
        mean_fitness = np.nanmean(all_fitness_histories, axis=0)
        std_fitness = np.nanstd(all_fitness_histories, axis=0)

        # Plot mean fitness with error bars
        generations = range(1, len(mean_fitness) + 1)
        plt.figure()
        plt.plot(generations, mean_fitness, label='Mean Fitness')
        plt.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.2)
        plt.title(f'Aggregated Fitness over Generations for Enemy {enemy}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.savefig(f"aggregated_fitness_enemy_{enemy}.png")
        plt.close()

        # Save aggregated data
        np.save(f"aggregated_mean_fitness_enemy_{enemy}.npy", mean_fitness)
        np.save(f"aggregated_std_fitness_enemy_{enemy}.npy", std_fitness)
    else:
        print(f"No data available for enemy {enemy}")

# Call the function for each enemy
for enemy in enemies:
    aggregate_fitness(enemy)
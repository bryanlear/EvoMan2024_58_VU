import numpy as np
import matplotlib.pyplot as plt

# Load gain data
gains_data_pso = np.load('gains_data_pso.npy', allow_pickle=True).item()
gains_data_ga = np.load('gains_data_ga.npy', allow_pickle=True).item()

# Create box plots for each enemy
enemies = [2, 5, 8]

for enemy in enemies:
    plt.figure(figsize=(8, 6))
    
    # Combine the data for PSO and GA
    data = [gains_data_pso[enemy], gains_data_ga[enemy]]
    
    # Create the box plot with labels for PSO and GA
    plt.boxplot(data, vert=True, patch_artist=True, labels=['PSO', 'GA'])

    plt.title(f'Individual Gain Box Plot - Enemy {enemy}')
    plt.ylabel('Individual Gain')
    plt.xlabel('Algorithm')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the box plot
    save_path = f'boxplot_enemy_{enemy}_comparison.png'
    plt.savefig(save_path)
    print(f'Saved box plot for Enemy {enemy} at: {save_path}')
    plt.show()
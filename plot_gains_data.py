import numpy as np
import matplotlib.pyplot as plt

# Load gain data for PSO
gains_data_pso = np.load('gains_data_pso.npy', allow_pickle=True).item()

# Create box plots for each enemy
enemies = [2, 5, 8]

for enemy in enemies:
    plt.figure(figsize=(8, 6))
    
    # Create the box plot with PSO data
    data = [gains_data_pso[enemy]]
    plt.boxplot(data, vert=True, patch_artist=True, labels=['PSO'])

    plt.title(f'Individual Gain Box Plot - Enemy {enemy}')
    plt.ylabel('Individual Gain')
    plt.xlabel('Algorithm (PSO)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the box plot
    plt.savefig(f'boxplot_enemy_{enemy}_pso.png')
    print(f'Saved box plot for Enemy {enemy} at: boxplot_enemy_{enemy}_pso.png')
    plt.show()
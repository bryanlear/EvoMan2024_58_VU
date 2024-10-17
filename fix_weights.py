import numpy as np
import os

num_runs = 10
enemies = [1, 2, 3, 4, 5, 6, 7, 8]  # Update this list to include all enemies you have data for

for enemy in enemies:
    for run in range(1, num_runs + 1):
        experiment_name = f"enemy_{enemy}_run_{run}"
        weights_file = f"{experiment_name}/best_weights_enemy_{enemy}_run_{run}.npy"

        # Check if the weights file exists
        if os.path.exists(weights_file):
            # Load with allow_pickle=True
            weights = np.load(weights_file, allow_pickle=True)
            
            # Check if weights is a 0-dimensional array (which contains an object)
            if weights.ndim == 0:
                # Extract the actual array from the object
                weights = weights.item()

            # Ensure weights is a NumPy array of type float64
            weights = np.array(weights, dtype=np.float64)
            
            # Resave the weights without pickling
            np.save(weights_file, weights)
            
            print(f"Processed weights for enemy {enemy}, run {run}")
        else:
            print(f"File {weights_file} not found.")
import numpy as np
import os

# Number of runs and enemies
num_runs = 10
enemies = [1, 2, 3, 4, 5, 6, 7, 8]

# Fitness weights (alpha, beta) for each enemy as per your original script
fitness_weights = {
    1: {'alpha': 1.5, 'beta': 1.0, 'gamma': 0.25},
    2: {'alpha': 1.0, 'beta': 1.2, 'gamma': 0.15},
    3: {'alpha': 1.5, 'beta': 1.3, 'gamma': 0.5},
    4: {'alpha': 1.0, 'beta': 1.8, 'gamma': 0.35},
    5: {'alpha': 1.2, 'beta': 1.2, 'gamma': 0.15},
    6: {'alpha': 1.4, 'beta': 1.3, 'gamma': 0.25},
    7: {'alpha': 1.0, 'beta': 1.5, 'gamma': 0.15},
    8: {'alpha': 1.0, 'beta': 1.7, 'gamma': 0.45},
}

# Directory where your experiment data is stored
# Adjust this path if necessary
base_dir = os.getcwd()  # Current working directory

for enemy in enemies:
    # Get alpha and beta for this enemy
    alpha = fitness_weights[enemy]['alpha']
    beta = fitness_weights[enemy]['beta']

    # Compute max adjusted fitness for this enemy
    max_adjusted_fitness = 100 * (alpha + beta)

    for run in range(1, num_runs + 1):
        # Construct the path to the fitness history file
        experiment_name = f"enemy_{enemy}_run_{run}"
        fitness_history_file = os.path.join(base_dir, experiment_name, f"fitness_history_enemy_{enemy}_run_{run}.npy")

        # Check if the fitness history file exists
        if os.path.exists(fitness_history_file):
            # Load the fitness history (negative values)
            fitness_history = np.load(fitness_history_file)

            # Convert fitness_history to a numpy array if it's not already
            fitness_history = np.array(fitness_history, dtype=np.float64)

            # Compute adjusted fitness (positive values)
            adjusted_fitness_history = -fitness_history  # Negate to make positive

            # Rescale the adjusted fitness values to a 0-100 scale
            scaled_fitness_history = (adjusted_fitness_history / max_adjusted_fitness) * 100

            # Ensure that scaled fitness values are within [0, 100]
            scaled_fitness_history = np.clip(scaled_fitness_history, 0, 100)

            # Save the rescaled fitness history to a new file
            rescaled_fitness_file = os.path.join(base_dir, experiment_name, f"scaled_fitness_history_enemy_{enemy}_run_{run}.npy")
            np.save(rescaled_fitness_file, scaled_fitness_history)

            print(f"Rescaled fitness values for Enemy {enemy}, Run {run} saved.")
        else:
            print(f"Fitness history file not found for Enemy {enemy}, Run {run}: {fitness_history_file}")
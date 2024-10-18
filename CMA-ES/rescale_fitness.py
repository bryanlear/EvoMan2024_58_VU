import os
import numpy as np
import matplotlib.pyplot as plt

# Root directory containing your runs (adjust if needed)
root_dir = '.'

# Lists to store all fitness values for global scaling
all_fitness_values = []

# First pass: Collect all fitness histories to determine global min and max
fitness_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.startswith('fitness_history_') and filename.endswith('.npy'):
            fitness_file = os.path.join(dirpath, filename)
            fitness_files.append(fitness_file)
            # Load fitness history
            fitness_history = np.load(fitness_file)
            all_fitness_values.extend(fitness_history)

# Convert to NumPy array
all_fitness_values = np.array(all_fitness_values)

# Compute global min and max fitness values
f_min = all_fitness_values.min()
f_max = all_fitness_values.max()

# Avoid division by zero if all fitness values are the same
if f_max - f_min == 0:
    print("All fitness values are the same. Cannot rescale.")
    exit()

# Second pass: Rescale and plot each fitness history
for fitness_file in fitness_files:
    # Load fitness history
    fitness_history = np.load(fitness_file)
    # Rescale fitness values to 0-100
    fitness_rescaled = (fitness_history - f_min) / (f_max - f_min) * 100

    # Save rescaled fitness history
    dirpath = os.path.dirname(fitness_file)
    filename = os.path.basename(fitness_file)
    rescaled_filename = 'scaled_' + filename
    rescaled_filepath = os.path.join(dirpath, rescaled_filename)
    np.save(rescaled_filepath, fitness_rescaled)

    # Generate generations list
    generations = np.arange(1, len(fitness_history) + 1)

    # Plot the rescaled fitness values
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_rescaled, marker='o', linestyle='-')
    plt.title(f'Rescaled Fitness over Generations\n{fitness_file}')
    plt.xlabel('Generation')
    plt.ylabel('Rescaled Fitness (0 = Worst, 100 = Best)')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_filename = filename.replace('fitness_history_', 'rescaled_fitness_plot_').replace('.npy', '.png')
    plot_filepath = os.path.join(dirpath, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    print(f"Processed and plotted {fitness_file}")

print("Rescaling and plotting completed.")
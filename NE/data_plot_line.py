

# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Set the file path
file_path = "NE/results_tests/NE_group1_results_.txt"
# Read the data file using pandas
# Adjust separator if needed (here using a comma)
data = pd.read_csv(file_path, sep=',')

# Set the column names (if needed)
data.columns = ["Run", "Generation", "Best_fitness", "Mean_fitness", "SD_mean_fitness"]

# Group the data by generation and calculate the mean and std deviation
best_fit_mean = data.groupby('Generation')['Best_fitness'].mean()
best_fit_sd = data.groupby('Generation')['Best_fitness'].std()

mean_fit_mean = data.groupby('Generation')['Mean_fitness'].mean()
mean_fit_sd = data.groupby('Generation')['Mean_fitness'].std()

# Create a dataframe for plotting
plot_data = pd.DataFrame({
    'Generation': best_fit_mean.index,
    'Best_Fitness_Mean': best_fit_mean.values,
    'Best_Fitness_SD': best_fit_sd.values,
    'Mean_Fitness_Mean': mean_fit_mean.values,
    'Mean_Fitness_SD': mean_fit_sd.values
})

# Plot the results
plt.figure(figsize=(10,6))

# Plot best fitness
plt.errorbar(plot_data['Generation'], plot_data['Best_Fitness_Mean'], yerr=plot_data['Best_Fitness_SD'], label='Best Fitness', fmt='-o')

# Plot mean fitness
plt.errorbar(plot_data['Generation'], plot_data['Mean_Fitness_Mean'], yerr=plot_data['Mean_Fitness_SD'], label='Mean Fitness', fmt='-o')

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Mean and Best Fitness Over Generations - Group 2')
plt.legend()

# Show the plot
plt.show()
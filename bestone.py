# Import necessary libraries
import pandas as pd
import numpy as np

# Set the file path

file_path = "NE/results_tests/Group1_final/NE_group1_results_.txt"
# Read the data file using pandas
# Adjust separator if needed (here using a comma)
data = pd.read_csv(file_path, sep=',')

# Set the column names (if needed)
data.columns = ["Run", "Generation", "Best_fitness", "Mean_fitness", "SD_mean_fitness"]

# Group the data by 'Run' to analyze the performance of each run
run_groups = data.groupby('Run')

# Dictionary to store run metrics
run_metrics = {}

# Iterate over each run to calculate metrics
for run, group in run_groups:
    max_best_fitness = group['Best_fitness'].max()  # Maximum best fitness in the run
    avg_best_fitness = group['Best_fitness'].mean()  # Average best fitness in the run
    avg_mean_fitness = group['Mean_fitness'].mean()  # Average mean fitness in the run
    
    # Store the metrics for this run
    run_metrics[run] = {
        'Max Best Fitness': max_best_fitness,
        'Avg Best Fitness': avg_best_fitness,
        'Avg Mean Fitness': avg_mean_fitness
    }

# Convert the dictionary to a DataFrame for easy viewing
metrics_df = pd.DataFrame.from_dict(run_metrics, orient='index')

# Find the best performing run based on max best fitness
best_run_max_fitness = metrics_df['Max Best Fitness'].idxmax()
best_run_avg_best_fitness = metrics_df['Avg Best Fitness'].idxmax()
best_run_avg_mean_fitness = metrics_df['Avg Mean Fitness'].idxmax()

print("Run Performance Metrics:")
print(metrics_df)

print(f"\nBest run based on maximum best fitness: Run {best_run_max_fitness}")
print(f"Best run based on average best fitness: Run {best_run_avg_best_fitness}")
print(f"Best run based on average mean fitness: Run {best_run_avg_mean_fitness}")
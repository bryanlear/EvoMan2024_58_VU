import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

# Parameters
n_hidden = 10  # Hidden neurons
num_runs = 10  # Number of runs for each group

# Function to test the best-trained model against all 8 enemies
def test_trained_model_against_all_enemies(weights, n_hidden):
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    fitness_scores = []  # To store fitness values for each enemy
    
    for enemy in enemies:
        try:
            # Set up environment for each enemy
            env = Environment(
                experiment_name=f"test_enemy_{enemy}",
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(n_hidden),
                enemymode="static",
                level=2,
                speed="fastest",
                multiplemode="no"
            )
            # Play the game and get fitness results
            f, p, e, t = env.play(pcont=weights)
            print(f"Test against Enemy {enemy}: Fitness = {f}, Player Health = {p}, Enemy Health = {e}, Time = {t}")
            
            # Append the fitness value for plotting
            fitness_scores.append(f)
        except Exception as ex:
            print(f"Error during test: {ex}")
            fitness_scores.append(None)  # In case of error, append None to keep the indexing consistent

    # Plot fitness results against all enemies
    plt.figure()
    plt.bar(enemies, fitness_scores, color='blue')
    plt.title('Fitness Scores Against All 8 Enemies')
    plt.xlabel('Enemy')
    plt.ylabel('Fitness')
    plt.xticks(enemies)
    plt.savefig(f"fitness_plot_all_enemies.png")
    plt.show()
    
    return fitness_scores

# Load and test the best models after training
def run_evaluation():
    enemy_groups = ['Group1', 'Group2']  # The groups used during training
    
    for group_name in enemy_groups:
        for run in range(1, num_runs + 1):
            best_weights_file = f"{group_name}_run_{run}/best_weights_{group_name}_run_{run}.npy"
            
            # Check if the file exists
            if os.path.exists(best_weights_file):
                best_weights = np.load(best_weights_file)
                print(f"\nTesting best model from {group_name} - Run {run} against all 8 enemies\n")
                
                # Test the best weights against all 8 enemies and plot fitness scores
                fitness_scores = test_trained_model_against_all_enemies(best_weights, n_hidden)
                
                # Save fitness scores to a file
                np.save(f"{group_name}_run_{run}/fitness_scores_all_enemies.npy", fitness_scores)
            else:
                print(f"Best weights file not found for {group_name} - Run {run}")

if __name__ == "__main__":
    run_evaluation()
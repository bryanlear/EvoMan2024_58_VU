import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# Load the best weights for a specific enemy
enemy = 8
best_weights = np.load(f"best_weights_enemy_{enemy}.npy")

# Create an environment with the specific enemy
env = Environment(experiment_name=f"evaluate_enemy_{enemy}",
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(10),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# Run the environment with the loaded best weights
fitness, player_health, enemy_health, time_spent = env.play(pcont=best_weights)

# Output the results
print(f"Fitness: {fitness}")
print(f"Player Health: {player_health}")
print(f"Enemy Health: {enemy_health}")
print(f"Time Spent: {time_spent}")
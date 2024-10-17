from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np

num_runs = 10
n_hidden = 10

def test_against_all_enemies(weights, n_hidden):
    gains = []
    player_life = []
    enemy_life = []
    times = []

    for enemy in range(1, 9):
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
        f, p, e, t = env.play(pcont=weights)
        gain = p - e

        gains.append(gain)
        player_life.append(p)
        enemy_life.append(e)
        times.append(t)

    return gains, player_life, enemy_life, times

enemies = [1,2,3,4,5,6,7,8]
# Load best weights from all runs and enemies
for enemy in enemies:
    all_gains = []
    all_player_life = []
    all_enemy_life = []
    for run in range(1, num_runs + 1):
        experiment_name = f"enemy_{enemy}_run_{run}"
        weights = np.load(f"{experiment_name}/best_weights_enemy_{enemy}_run_{run}.npy")
        gains, player_life, enemy_life, times = test_against_all_enemies(weights, n_hidden)
        
        print(f"Weights type: {type(weights)}")
        print(f"Weights shape: {weights.shape}")

        # Store results
        all_gains.append(gains)
        all_player_life.append(player_life)
        all_enemy_life.append(enemy_life)

    # Save the results for statistical analysis
    np.save(f"test_results_enemy_{enemy}_gains.npy", all_gains)
    np.save(f"test_results_enemy_{enemy}_player_life.npy", all_player_life)
    np.save(f"test_results_enemy_{enemy}_enemy_life.npy", all_enemy_life)
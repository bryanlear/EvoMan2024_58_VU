import matplotlib.pyplot as plt
import numpy as np

enemies = [1, 2, 3, 4, 5, 6, 7, 8]


def generate_box_plots(enemy):
    gains = np.load(f"test_results_enemy_{enemy}_gains.npy")
    player_life = np.load(f"test_results_enemy_{enemy}_player_life.npy")
    enemy_life = np.load(f"test_results_enemy_{enemy}_enemy_life.npy")

    # Transpose to get results per enemy
    gains = np.array(gains).T
    player_life = np.array(player_life).T
    enemy_life = np.array(enemy_life).T

    # Box plot for gains against all enemies
    plt.figure()
    plt.boxplot(gains)
    plt.title(f'Gains Against All Enemies - Best of Enemy {enemy}')
    plt.xlabel('Enemy')
    plt.ylabel('Gain')
    plt.savefig(f"boxplot_gains_enemy_{enemy}.png")
    plt.close()

    # Similarly for player life
    plt.figure()
    plt.boxplot(player_life)
    plt.title(f'Player Life Against All Enemies - Best of Enemy {enemy}')
    plt.xlabel('Enemy')
    plt.ylabel('Player Life')
    plt.savefig(f"boxplot_player_life_enemy_{enemy}.png")
    plt.close()

# Generate box plots for each enemy
for enemy in enemies:
    generate_box_plots(enemy)
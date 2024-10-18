from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
from cma import CMAEvolutionStrategy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import pickle

# Number of independent runs
num_runs = 10

# Parameters
n_hidden = 10  # Hidden neurons

# Define enemy groups for training
enemy_groups = {
    'Group2': [2, 4, 6, 7]
}

# Adjustments for sigma, population sizes, and fitness weights
sigma_values = {
    1: 0.75,
    2: 0.85,
    3: 0.8,
    4: 0.5,
    5: 0.8,
    6: 0.9,
    7: 1.1,
    8: 0.5,
}

population_sizes = {
    1: 200,
    2: 200,
    3: 200,
    4: 200,
    5: 200,
    6: 200,
    7: 200,
    8: 200,
}

n_generations_dict = {
    1: 40,
    2: 40,
    3: 30,
    4: 30,
    5: 30,
    6: 30,
    7: 30,
    8: 30
}

# Adjust the fitness weights based on previous results
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

# Parameters for adaptive sigma
sigma_adjustment_window = 5
sigma_adjustment_factor = {
    1: 0.85,
    2: 0.8,
    3: 0.9,
    4: 0.8,
    5: 0.85,
    6: 0.9,
    7: 0.8,
    8: 0.75,
}
stagnation_threshold = 2  # Number of windows with no improvement for quicker adaptation

# Evaluation function for enemy groups
def evaluate_group_individual(args):
    weights, enemy_group, n_hidden = args
    total_fitness = 0
    total_gain = 0
    total_damage_to_enemy = 0
    total_time_penalty = 0

    for enemy in enemy_group:
        try:
            env = Environment(
                experiment_name=f"group_{enemy}_experiment",
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(n_hidden),
                enemymode="static",
                level=2,
                speed="fastest",
                multiplemode="no"
            )
            f, p, e, t = env.play(pcont=weights)
        except Exception as ex:
            print(f"Error during simulation: {ex}")
            f, p, e, t = -np.inf, 0, 0, 0

        gain = p - e
        damage_to_enemy = 100 - e
        time_penalty = t / 500.0 * (1 + (100 - e) / 100)

        # Get the enemy-specific alpha, beta, and gamma
        alpha = fitness_weights[enemy]['alpha']
        beta = fitness_weights[enemy]['beta']
        gamma = fitness_weights[enemy]['gamma']

        # Calculate fitness for this enemy
        fitness = - (alpha * gain + beta * damage_to_enemy - gamma * time_penalty)

        # Accumulate for the group
        total_fitness += fitness
        total_gain += gain
        total_damage_to_enemy += damage_to_enemy
        total_time_penalty += time_penalty

    # Return average for the group
    return total_fitness / len(enemy_group), total_gain / len(enemy_group), total_damage_to_enemy / len(enemy_group), total_time_penalty / len(enemy_group)

# Main block for group evaluation
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Use 'spawn' for macOS

    for group_name, group_enemies in enemy_groups.items():
        for run in range(1, num_runs + 1):
            print(f"Starting run {run}/{num_runs} for {group_name}")

            # Set random seed
            np.random.seed(run * 42)  # Multiplying by run to change the seed each time

            # Modify experiment_name to include the run number
            experiment_name = f"{group_name}_run_{run}"
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)

            # Set up environment (using the first enemy just for environment setup)
            env = Environment(
                experiment_name=experiment_name,
                enemies=[group_enemies[0]],
                playermode="ai",
                player_controller=player_controller(n_hidden),
                enemymode="static",
                level=2,
                speed="fastest",
                multiplemode="no"
            )

            n_inputs = env.get_num_sensors()
            n_vars = (n_inputs + 1) * n_hidden + (n_hidden + 1) * 5

            # Group-specific parameters (taking the first enemy's sigma for initialization)
            sigma = sigma_values[group_enemies[0]]
            population_size = population_sizes[group_enemies[0]]
            n_generations = n_generations_dict[group_enemies[0]]

            # Initialize CMA-ES
            x0 = np.random.randn(n_vars)
            es = CMAEvolutionStrategy(x0, sigma, {'popsize': population_size})
            
                        # Fitness history tracking
            fitness_history = []
            gains_history = []
            damages_history = []
            time_penalties_history = []

            for generation in range(n_generations):
                solutions = es.ask()
                args = [(weights, group_enemies, n_hidden) for weights in solutions]

                # Multiprocessing evaluation for the group
                with Pool(processes=2) as pool:
                    results = pool.map(evaluate_group_individual, args)

                fitnesses, gains, damages_to_enemy, time_penalties = zip(*results)

                es.tell(solutions, fitnesses)
                es.disp()

                # Track best fitness for the group
                best_fitness = min(fitnesses)
                fitness_history.append(best_fitness)
                gains_history.append(np.mean(gains))
                damages_history.append(np.mean(damages_to_enemy))
                time_penalties_history.append(np.mean(time_penalties))

                # Compute average fitness components
                avg_gain = np.mean(gains)
                avg_damage = np.mean(damages_to_enemy)
                avg_time_penalty = np.mean(time_penalties)

                print(f"Generation {generation + 1}/{n_generations} - Best Fitness: {best_fitness}")
                print(f"Average Gain: {avg_gain}, Average Damage: {avg_damage}, Average Time Penalty: {avg_time_penalty}")

                # Adaptive sigma adjustment for group
                if (generation + 1) % sigma_adjustment_window == 0:
                    if len(fitness_history) >= sigma_adjustment_window:
                        if np.abs(fitness_history[-sigma_adjustment_window] - best_fitness) < 1e-3:
                            # No significant improvement --> reduce sigma more aggressively
                            es.sigma *= sigma_adjustment_factor[group_enemies[0]]
                            print(f"Reducing sigma to {es.sigma} due to stagnation.")

                    # If stagnation continues over multiple windows
                    if len(fitness_history) >= sigma_adjustment_window * stagnation_threshold:
                        recent_history = fitness_history[-sigma_adjustment_window * stagnation_threshold:]
                        if np.std(recent_history) < 1e-3:
                            # Reinitialize CMA-ES with new random population
                            es = CMAEvolutionStrategy(x0, es.sigma, {'popsize': population_size})
                            print("Reinitializing CMA-ES due to prolonged stagnation.")
                            fitness_history = []

            # Retrieve the best solution from CMA-ES for the group
            best_solution = es.result.xbest
            # Save the best solution for each run
            np.save(f"{experiment_name}/best_weights_{group_name}_run_{run}.npy", best_solution)

            # Save fitness history and components for the group
            np.save(f"{experiment_name}/fitness_history_{group_name}_run_{run}.npy", fitness_history)
            np.save(f"{experiment_name}/gains_history_{group_name}_run_{run}.npy", gains_history)
            np.save(f"{experiment_name}/damages_history_{group_name}_run_{run}.npy", damages_history)
            np.save(f"{experiment_name}/time_penalties_history_{group_name}_run_{run}.npy", time_penalties_history)

            # Visualization
            plt.figure()
            plt.plot(fitness_history)
            plt.title(f'Fitness over Generations for {group_name} - Run {run}')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.savefig(f"{experiment_name}/fitness_plot_{group_name}_run_{run}.png")
            plt.close()

            # Plot fitness components
            plt.figure()
            plt.plot(gains_history, label='Average Gain')
            plt.plot(damages_history, label='Average Damage')
            plt.plot(time_penalties_history, label='Average Time Penalty')
            plt.title(f'Fitness Components over Generations for {group_name} - Run {run}')
            plt.xlabel('Generation')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f"{experiment_name}/fitness_components_{group_name}_run_{run}.png")
            plt.close()
            
            
################################################
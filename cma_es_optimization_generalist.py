from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
from cma import CMAEvolutionStrategy
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

# Random seed - reproducibility
np.random.seed(42)

# Parameters
n_hidden = 10  # Hidden neurons

# Enemy list
enemies = [1, 2, 3, 4, 5, 6, 7, 8]

# Adjustments for sigma, population sizes, and fitness weights
sigma_values = {
    1: 0.75,  
    2: 0.85,  
    3: 0.8,   
    4: 0.5,  
    5: 0.8,   
    6: 0.9,   
    7: 1.2,   
    8: 0.5,   
}

population_sizes = {
    1: 800,  
    2: 600,  
    3: 600,   
    4: 500,  
    5: 500,  
    6: 600,  
    7: 500,  
    8: 800, 
}

n_generations_dict = {
    1: 70,  
    2: 100,  
    3: 55,  
    4: 35,  
    5: 50,  
    6: 70,  
    7: 70,  
    8: 100  
}

# Adjust the fitness weights based on previous results
fitness_weights = {
    1: {'alpha': 1.5, 'beta': 1.0, 'gamma': 0.25},  # Less emphasis on time penalty
    2: {'alpha': 1.0, 'beta': 1.2, 'gamma': 0.15},  # Reduced time penalty weight
    3: {'alpha': 1.5, 'beta': 1.3, 'gamma': 0.5},  
    4: {'alpha': 1.0, 'beta': 1.8, 'gamma': 0.35}, 
    5: {'alpha': 1.2, 'beta': 1.2, 'gamma': 0.15}, 
    6: {'alpha': 1.4, 'beta': 1.3, 'gamma': 0.25},  # Adjusted for better convergence
    7: {'alpha': 1.0, 'beta': 1.5, 'gamma': 0.15}, # Reduced time penalty
    8: {'alpha': 1.0, 'beta': 1.7, 'gamma': 0.45},  # Better time management
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

# Evaluation function at module level
def evaluate_individual(args):
    weights, enemy, n_hidden = args
    try:
        env = Environment(
            experiment_name=f"enemy_{enemy}_experiment",
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

    # Calculate fitness
    fitness = - (alpha * gain + beta * damage_to_enemy - gamma * time_penalty)

    return fitness, gain, damage_to_enemy, time_penalty

# Multiprocessing protection
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Use 'spawn' for macOS

    for enemy in enemies:
        print(f"Starting evolution for Enemy {enemy}")

        experiment_name = f"enemy_{enemy}_experiment"
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(n_hidden),
            enemymode="static",
            level=2,
            speed="fastest",
            multiplemode="no"
        )

        n_inputs = env.get_num_sensors()
        n_vars = (n_inputs + 1) * n_hidden + (n_hidden + 1) * 5

        # Enemy-specific parameters
        sigma = sigma_values[enemy]
        population_size = population_sizes[enemy]
        n_generations = n_generations_dict[enemy]

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
            args = [(weights, enemy, n_hidden) for weights in solutions]

            # Multiprocessing evaluation
            with Pool(processes=4) as pool:
                results = pool.map(evaluate_individual, args)

            fitnesses, gains, damages_to_enemy, time_penalties = zip(*results)

            es.tell(solutions, fitnesses)
            es.disp()

            # Track best fitness
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

            # Adaptive sigma adjustment for each enemy based on more aggressive decay for specific ones
            if (generation + 1) % sigma_adjustment_window == 0:
                if len(fitness_history) >= sigma_adjustment_window:
                    if np.abs(fitness_history[-sigma_adjustment_window] - best_fitness) < 1e-3:
                        # No significant improvement --> reduce sigma more aggressively
                        es.sigma *= sigma_adjustment_factor[enemy]
                        print(f"Reducing sigma to {es.sigma} due to stagnation.")

                # If stagnation continues over multiple windows
                if len(fitness_history) >= sigma_adjustment_window * stagnation_threshold:
                    recent_history = fitness_history[-sigma_adjustment_window * stagnation_threshold:]
                    if np.std(recent_history) < 1e-3:
                        # Reinitialize CMA-ES with new random population
                        es = CMAEvolutionStrategy(x0, es.sigma, {'popsize': population_size})
                        print("Reinitializing CMA-ES due to prolonged stagnation.")
                        fitness_history = []

        # Save the best solution for enemy
        best_solution = es.result.xbest
        np.save(f"{experiment_name}/best_weights_enemy_{enemy}.npy", best_solution)

        # Save fitness history and components
        np.save(f"{experiment_name}/fitness_history_enemy_{enemy}.npy", fitness_history)
        np.save(f"{experiment_name}/gains_history_enemy_{enemy}.npy", gains_history)
        np.save(f"{experiment_name}/damages_history_enemy_{enemy}.npy", damages_history)
        np.save(f"{experiment_name}/time_penalties_history_enemy_{enemy}.npy", time_penalties_history)

        # Visualization
        plt.figure()
        plt.plot(fitness_history)
        plt.title(f'Fitness over Generations for Enemy {enemy}')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.savefig(f"{experiment_name}/fitness_plot_enemy_{enemy}.png")
        plt.close()

        # Plot fitness components
        plt.figure()
        plt.plot(gains_history, label='Average Gain')
        plt.plot(damages_history, label='Average Damage')
        plt.plot(time_penalties_history, label='Average Time Penalty')
        plt.title(f'Fitness Components over Generations for Enemy {enemy}')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f"{experiment_name}/fitness_components_enemy_{enemy}.png")
        plt.close()
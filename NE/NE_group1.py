# Basics
import sys, os
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = "NE_group1"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Create a run without the visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Train model or test best individuals
mode = "test"


# Parameters
best_individual_retries = 5
population_size = survivors = 100 #100
gens = 30 #30
runs = 10 #10


input_size = 20
hidden_neurons_size = 10
output_size = 5
total_genes = (input_size + 1) * hidden_neurons_size + (hidden_neurons_size + 1) * output_size

crossover_probability = 0.7 #0.7
tournament_size = 5
mutation_probability = 0.2 #0.2


# Environment
env = Environment(experiment_name = experiment_name,
                  enemies = [2, 4, 6, 7], 
                  multiplemode = "yes",
                  playermode = "ai",
                  player_controller = player_controller(hidden_neurons_size),
                  enemymode = "static",
                  level = 2,
                  speed = "fastest",
                  visuals = False)


# Simulation
def simulation(env, individual_x):
    fitness, player_health, enemy_health, duration = env.play(pcont=individual_x)
    if mode == "train":
        return fitness
    elif mode == "test":
        return fitness, player_health, enemy_health


# Evaluation
def evaluation(current_gen):
    return np.array(list(map(lambda individual_y: simulation(env, individual_y), current_gen)))


# Parents selection (tournament) --
def select_parents(current_gen, tournament_participants):
    # Random selection from current generation
    participants = np.random.choice(current_gen.shape[0], tournament_participants, replace=False)
    selected_parents = participants[np.argmax(population_fitness_scores[participants])]

    return current_gen[selected_parents]


# Mutation and Crossover
def produce_offspring(current_gen):
    number_offspring = np.zeros((0, total_genes))

    for parent in range(current_gen.shape[0]):
        parent_1 = select_parents(current_gen, tournament_size)
        parent_2 = select_parents(current_gen, tournament_size)

        if np.random.uniform(0, 1) <= crossover_probability:
            crossover_points = sorted(np.random.choice(total_genes, 3, replace=False))
            child1 = np.copy(parent_1)
            child1[crossover_points[0]:crossover_points[1]] = parent_2[crossover_points[0]:crossover_points[1]]
            child1[crossover_points[2]:] = parent_2[crossover_points[2]:]

            for i in range(len(child1)):
                if np.random.uniform(0, 1) <= mutation_probability:
                    child1[i] += np.random.normal(0, .5)

            for i in range(len(child1)):
                if child1[i] > 1:
                    child1[i] = 1
                elif child1[i] < -1:
                    child1[i] = -1

            child2 = np.copy(parent_2)
            child2[crossover_points[0]:crossover_points[1]] = parent_1[crossover_points[0]:crossover_points[1]]
            child2[crossover_points[2]:] = parent_1[crossover_points[2]:]

            for i in range(len(child2)):
                if np.random.uniform(0, 1) <= mutation_probability:
                    child2[i] += np.random.normal(0, .5)

            for i in range(len(child2)):
                if child2[i] > 1:
                    child2[i] = 1
                elif child2[i] < -1:
                    child2[i] = -1

            number_offspring = np.vstack((number_offspring, child1))
            number_offspring = np.vstack((number_offspring, child2))

    return number_offspring


# Select survivors
def select_survivors(current_gen_parents_offspring, fitness_current_gen_parents_offspring):
    order = np.argsort(fitness_current_gen_parents_offspring)
    selected_survivors = current_gen_parents_offspring[order[-survivors:]]

    return selected_survivors


# Test best solutions
if mode == "test":
    env = Environment(experiment_name = experiment_name,
                      enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(hidden_neurons_size),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    for run in range(1, runs+1):
        best_individual_filename = experiment_name + f"/NE_group1_best_individual_{run}.txt"
        best = np.loadtxt(best_individual_filename)

        print(f"Best individual in run {run}\n")

        for repeat in range(1, best_individual_retries+1):
            results = evaluation([best])
            enemy_health = results[0][2]
            player_health = results[0][1]


            gain = player_health - enemy_health
            print(f"Gain {gain} \n")
            best_individual_retries_file = experiment_name + f"/NE_group1_best_individual_retries.txt"
            results_file = open(best_individual_retries_file, 'a')
            results_file.write(f"{run},{repeat},{gain}\n")
            results_file.close()


# Evolution 
elif mode == "train":
    for run in range(1, runs + 1):
        for gen in range(gens):
            print(f"Run {run} \nGeneration {gen + 1}/{gens}")

            if gen == 0:
                population = np.random.uniform(-1, 1, (population_size, total_genes))

            else:
                population = new_population

            population_fitness_scores = evaluation(population)
            best_fitness_scores = np.argmax(population_fitness_scores)
            avg_fitness_scores = np.mean(population_fitness_scores)
            fitness_std_dev = np.std(population_fitness_scores)
            print(f" Best fitness score: {population_fitness_scores[best_fitness_scores]} \n Mean fitness score: {avg_fitness_scores} \n Standard deviation: {fitness_std_dev}\n")

            results_file = open(experiment_name + '/NE_group1_results_.txt', 'a')
            results_file.write(f"{run},{gen + 1},{population_fitness_scores[best_fitness_scores]},{avg_fitness_scores},{fitness_std_dev}\n")
            results_file.close()

            best_individual_filename = experiment_name + f"/NE_group1_best_individual_{run}.txt"
            np.savetxt(best_individual_filename, population[best_fitness_scores])

            offspring = produce_offspring(population)
            fitness_offspring = evaluation(offspring)

            parents_and_offspring = np.vstack((population, offspring))
            fitness_parents_and_offspring = np.append(population_fitness_scores, fitness_offspring)

            selection = select_survivors(parents_and_offspring, fitness_parents_and_offspring)
            new_population = selection


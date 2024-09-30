import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from evoman.environment import Environment

n_runs = 10
enemies = [2, 5, 8]

# Load the gain data
gains_data = np.load('gains_data_pso.npy', allow_pickle=True).item()

#print(gains_data)

# Access enemy data
enemy_3_gains = gains_data[3]
print(f"Gains for Enemy 3 across 10 runs: {enemy_3_gains}")

enemy_6_gains = gains_data[6]
print(f"Gains for Enemy 6 across 10 runs: {enemy_6_gains}")

enemy_8_gains = gains_data[8]
print(f"Gains for Enemy 8 across 10 runs: {enemy_8_gains}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
 # load them data
gains_data_pso = np.load('gains_data_pso.npy', allow_pickle=True).item()
gains_data_ga = np.load('gains_data_ga.npy', allow_pickle=True).item()

# which enemies you fighting, homie ?
enemies = [2, 5, 8]

#pre-
combined_data = []
for enemy in enemies:
    # Combine PSO and GA data for each enemy
    for value in gains_data_pso[enemy]:
        combined_data.append({"Algorithm": "PSO", "Gain": value, "Enemy": f"Enemy {enemy}"})
    for value in gains_data_ga[enemy]:
        combined_data.append({"Algorithm": "GA", "Gain": value, "Enemy": f"Enemy {enemy}"})


df = pd.DataFrame(combined_data)
plt.figure(figsize=(12, 6))
sns.boxplot(x="Enemy", y="Gain", hue="Algorithm", data=df, palette="Set3", width=0.6)

# Perform t-tests and get the Ps
for i, enemy in enumerate(enemies):
    # Extract data for t-test
    pso_data = gains_data_pso[enemy]
    ga_data = gains_data_ga[enemy]

    # Perform t-test
    t_stat, p_value = ttest_ind(pso_data, ga_data, equal_var=False)


    y_max = max(max(pso_data), max(ga_data)) + 5  
    x_pos = i 

    # Annotate p-value
    plt.text(x=x_pos, y=y_max, s=f"p = {p_value:.3f}", ha='center', fontsize=10, fontweight='bold')


plt.title("Individual Gain Comparison between PSO and GA")
plt.xlabel("Enemy")
plt.ylabel("Individual Gain")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and display
plt.savefig('comparison_boxplots_with_p_values.png')
plt.show()
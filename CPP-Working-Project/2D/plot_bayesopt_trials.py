import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file using numpy (skip header)
csv_path = "../2D/2D_bayesopt_trials.csv"
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

# Filter out invalid or penalized rows
valid = np.isfinite(data[:, 2]) & (data[:, 2] < 1e5)
Q = data[valid, 0]
R = data[valid, 1]
C = data[valid, 2]

plt.figure(figsize=(8, 6))
sc = plt.scatter(Q, R, c=C, cmap='viridis', s=60, edgecolor='k')
plt.xlabel('Q (Process Noise Diagonal)')
plt.ylabel('R (Measurement Noise Diagonal)')
plt.title('BayesOpt Trials: Q, R vs. Consistency Metric C')
cbar = plt.colorbar(sc)
cbar.set_label('C (Consistency Metric)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() 
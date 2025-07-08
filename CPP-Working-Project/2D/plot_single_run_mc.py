import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "2D_single_run_mc_results.csv"
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
chi2 = data[:, 1]
mse = data[:, 2]

# Sort by chi2 (ascending)
sort_idx = np.argsort(mse)
chi2_sorted = chi2[sort_idx]
mse_sorted = mse[sort_idx]
runs_sorted = np.arange(len(chi2_sorted))

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
l1 = ax1.plot(runs_sorted, chi2_sorted, 'o-', color=color1, label='Chi² (sorted)')
ax1.set_xlabel('Sorted Run Index (by Chi²)')
ax1.set_ylabel('Chi²', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
l2 = ax2.plot(runs_sorted, mse_sorted, 's--', color=color2, label='MSE (matched)')
ax2.set_ylabel('MSE', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends
lns = l1 + l2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right')

plt.title('Chi² (sorted) and Matched MSE per Run')
plt.tight_layout()
plt.show() 
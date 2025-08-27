import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import yaml
import os

# Load the HDF5 file
h5_path = "../2D-Tracking/Saved_Data/2D_bayesopt_trials.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['trials'][:]

# Filter out invalid or penalized rows
valid = np.isfinite(data[:, 2]) & (data[:, 2] < 1e5)
Q = data[valid, 0]
R = data[valid, 1]
C = data[valid, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter all points as before
sc = ax.scatter(Q, R, C, c=C, cmap='viridis', s=60, edgecolor='k')

# Find the minimum C value and threshold
min_C = np.min(C)
threshold = min_C + 0.1

# Highlight points within 0.05 of the minimum C value
close_mask = C <= threshold
ax.scatter(Q[close_mask], R[close_mask], C[close_mask], color='red', s=80, edgecolor='k', label='Within 0.05 of min(C)')

ax.set_xlabel('Q (Process Noise Diagonal)')
ax.set_ylabel('R (Measurement Noise Diagonal)')
ax.set_zlabel('C (Consistency Metric)')
ax.set_title('BayesOpt Trials: Q, R, C (3D)')
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('C (Consistency Metric)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()

# 2D plot of highlighted points (Q vs R) with line of best fit
Q_close = Q[close_mask]
R_close = R[close_mask]

plt.figure(figsize=(8, 6))
plt.scatter(Q_close, R_close, color='red', s=80, edgecolor='k', label='Low C Points')

# Fit a line of best fit (least squares)
if len(Q_close) > 1:
    coeffs = np.polyfit(Q_close, R_close, 1)  # Linear fit
    Q_fit = np.linspace(0, 1, 100)  # Cover the full axis range
    R_fit = np.polyval(coeffs, Q_fit)
    plt.plot(Q_fit, R_fit, color='blue', linewidth=2, label='Best Fit Line')

plt.xlabel('Q (Process Noise Diagonal)')
plt.ylabel('R (Measurement Noise Diagonal)')
plt.title('Low C Points with Best Fit Line')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Added: filtered 2D plot around true (q, R)
try:
    with open('../scenario_linear.yaml', 'r') as yf:
        cfg = yaml.safe_load(yf)
    q_true = float(cfg['Data_Generation']['q'])
    R_true = float(cfg['Data_Generation']['meas_noise_var'])
    abs_tol = float(os.environ.get('BO_FILTER_ABS_TOL', '0.2'))  # absolute window, default 0.001
    q_min, q_max = q_true - abs_tol, q_true + abs_tol
    R_min, R_max = R_true - abs_tol, R_true + abs_tol

    mask = (Q >= q_min) & (Q <= q_max) & (R >= R_min) & (R <= R_max)
    Q_f = Q[mask]
    R_f = R[mask]
    C_f = C[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_f, R_f, c=C_f, cmap='viridis', s=60, edgecolor='k')
    plt.colorbar(label='C (Consistency Metric)')
    plt.scatter([q_true], [R_true], color='red', s=120, marker='*', label='True (q, R)')
    plt.xlabel('Q (Process Noise Diagonal)')
    plt.ylabel('R (Measurement Noise Diagonal)')
    plt.title(f'Filtered Q–R near true values (±{abs_tol})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Skipped filtered 2D plot: {e}") 
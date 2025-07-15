import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import os

# Ask user for 1D, 2D, or MC results
choice = input('Show results for (1) 1D, (2) 2D, or (3) MC Results? Enter 1, 2, or 3: ').strip()
if choice == '1':
    h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/1D/1d_trajectory_estimate.h5'
    is_1d = True
    is_mc = False
elif choice == '2':
    h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2D/2D_trajectory_estimate.h5'
    is_1d = False
    is_mc = False
else:
    h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2D/2D_single_run_mc_results.h5'
    is_mc = True
    is_1d = False

def load_h5_data(path, is_1d):
    with h5py.File(path, 'r') as f:
        data = f['trajectory'][:]
        if is_1d:
            # Columns: t, true_x, meas_x, est_x
            t = data[:, 0]
            x_true = data[:, 1]
            x_meas = data[:, 2]
            x_est = data[:, 3]
            return t, x_true, x_meas, x_est
        else:
            # Columns: t, true_x, true_y, meas_x, meas_y, est_x, est_y
            t = data[:, 0]
            x_true = data[:, 1]
            y_true = data[:, 2]
            x_meas = data[:, 3]
            y_meas = data[:, 4]
            x_est = data[:, 5]
            y_est = data[:, 6]
            return t, x_true, y_true, x_meas, y_meas, x_est, y_est

def load_mc_results(path):
    with h5py.File(path, 'r') as f:
        data = f['results'][:]
        # Columns: run, chi2, mse
        run = data[:, 0]
        chi2 = data[:, 1]
        mse = data[:, 2]
        return run, chi2, mse

if is_mc:
    run, chi2, mse = load_mc_results(h5_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(run, chi2, 'o-', label='chi2')
    plt.xlabel('Run')
    plt.ylabel('chi2')
    plt.title('Chi2 per Monte Carlo Run')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(run, mse, 'o-', label='MSE', color='orange')
    plt.xlabel('Run')
    plt.ylabel('MSE')
    plt.title('MSE per Monte Carlo Run')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
elif is_1d:
    t, x_true, x_meas, x_est = load_h5_data(h5_path, is_1d=True)
    norm = plt.Normalize(t.min(), t.max())
    cmap = cm.viridis
    plt.figure(figsize=(10, 6))
    for i in range(len(t)):
        plt.scatter(x_true[i], 0, color=cmap(norm(t[i])), marker='o', label='True x' if i == 0 else "", s=60)
        plt.scatter(x_meas[i], 1, color=cmap(norm(t[i])), marker='s', label='Measured x' if i == 0 else "", s=60, alpha=0.7)
        plt.scatter(x_est[i], 2, color=cmap(norm(t[i])), marker='x', label='Estimated x' if i == 0 else "", s=60)
    x_min = min(np.min(x_true), np.min(x_meas), np.min(x_est))
    x_max = max(np.max(x_true), np.max(x_meas), np.max(x_est))
    for x in range(int(np.floor(x_min)), int(np.ceil(x_max)) + 1):
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Time step (t)')
    plt.yticks([0, 1, 2], ['True', 'Measured', 'Estimated'])
    plt.xlabel('State value (x)')
    plt.ylabel('Type')
    plt.title('1D State: True vs Measured vs Estimated (Color = Time Step)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
else:
    t, x_true, y_true, x_meas, y_meas, x_est, y_est = load_h5_data(h5_path, is_1d=False)
    plt.figure(figsize=(10, 6))
    plt.plot(x_true, y_true, 'o-', label='True Trajectory', color='green', markersize=5)
    plt.plot(x_meas, y_meas, 'o', label='Measured Trajectory', color='orange', markersize=5)
    plt.plot(x_est, y_est, 'x-', label='Estimated Trajectory', color='blue', markersize=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Trajectory: True vs Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


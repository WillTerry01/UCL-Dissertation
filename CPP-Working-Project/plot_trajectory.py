import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Ask user for 1D or 2D
choice = input('Show results for (1) 1D or (2) 2D? Enter 1 or 2: ').strip()
if choice == '1':
    csv_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/1D/1d_trajectory_estimate.csv'
    is_1d = True
else:
    csv_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2D/2D_trajectory_estimate.csv'
    is_1d = False

def load_csv_skip_chi2(path):
    # Read all lines
    with open(path, 'r') as f:
        lines = f.readlines()
    # Remove chi2 row if present
    lines = [line for line in lines if not line.startswith('chi2')]
    # Save to a temporary file-like object for numpy
    from io import StringIO
    data_str = ''.join(lines)
    data = np.genfromtxt(StringIO(data_str), delimiter=',', skip_header=1)
    return data

data = load_csv_skip_chi2(csv_path)

if is_1d:
    # Columns: t,true_x,meas_x,est_x
    t = data[:, 0]
    x_true = data[:, 1]
    x_meas = data[:, 2]
    x_est = data[:, 3]
    norm = plt.Normalize(t.min(), t.max())
    cmap = cm.viridis
    plt.figure(figsize=(10, 6))
    # Plot each point with color by time step, y=0 for True, y=1 for Measured, y=2 for Estimated
    for i in range(len(t)):
        plt.scatter(x_true[i], 0, color=cmap(norm(t[i])), marker='o', label='True x' if i == 0 else "", s=60)
        plt.scatter(x_meas[i], 1, color=cmap(norm(t[i])), marker='s', label='Measured x' if i == 0 else "", s=60, alpha=0.7)
        plt.scatter(x_est[i], 2, color=cmap(norm(t[i])), marker='x', label='Estimated x' if i == 0 else "", s=60)
    # Add vertical grid lines at every integer x value
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
    # Columns: t,true_x,true_y,meas_x,meas_y,est_x,est_y
    t = data[:, 0]
    x_true = data[:, 1]
    y_true = data[:, 2]
    x_meas = data[:, 3]
    y_meas = data[:, 4]
    x_est = data[:, 5]
    y_est = data[:, 6]
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


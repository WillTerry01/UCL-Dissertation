import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file using numpy (skip header, handle chi2 row)
csv_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2d_trajectory_estimate.csv'

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

# Columns: t,true_x,true_y,meas_x,meas_y,est_x,est_y
t = data[:, 0]
x_true = data[:, 1]
y_true = data[:, 2]
x_meas = data[:, 3]
y_meas = data[:, 4]
x_est = data[:, 5]
y_est = data[:, 6]

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(x_true, y_true, 'o-', label='True Trajectory', color='green', markersize=5)
plt.plot(x_meas, y_meas, 's-', label='Measured Trajectory', color='orange', markersize=5)
plt.plot(x_est, y_est, 'x-', label='Estimated Trajectory', color='blue', markersize=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Trajectory: True vs Measured vs Estimated')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


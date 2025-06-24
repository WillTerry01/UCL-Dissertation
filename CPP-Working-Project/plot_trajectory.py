import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2d_trajectory_estimate.csv'
data = pd.read_csv(csv_path)
data.columns = data.columns.str.strip()

print(data.head())
print(data.columns)

# Prepare data for plotting
x_true = data['true_x'].values
y_true = data['true_y'].values
x_est = data['est_x'].values
y_est = data['est_y'].values
t = data['t'].values

plt.figure(figsize=(10, 5))

# Plot true trajectory with color gradient
sc1 = plt.scatter(x_true, y_true, c=t, cmap='viridis', label='True Trajectory', marker='o', s=30)
# Plot estimated trajectory with color gradient
sc2 = plt.scatter(x_est, y_est, c=t, cmap='plasma', label='Estimated Trajectory', marker='x', s=30)

plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Trajectory: True vs Estimated (Color = Time Step)')
plt.legend(['True Trajectory', 'Estimated Trajectory'])
cbar = plt.colorbar(sc1, label='Time step')
plt.tight_layout()
plt.show()

# Plot velocity vectors (vx vs vy) with color gradient for time
plt.figure(figsize=(10, 5))

vx_true = data['true_vx'].values
vy_true = data['true_vy'].values
vx_est = data['est_vx'].values
vy_est = data['est_vy'].values

sc3 = plt.scatter(vx_true, vy_true, c=t, cmap='viridis', label='True Velocity', marker='o', s=30)
sc4 = plt.scatter(vx_est, vy_est, c=t, cmap='plasma', label='Estimated Velocity', marker='x', s=30)

plt.xlabel('vx')
plt.ylabel('vy')
plt.title('2D Velocity: True vs Estimated (Color = Time Step)')
plt.legend(['True Velocity', 'Estimated Velocity'])
cbar2 = plt.colorbar(sc3, label='Time step')
plt.tight_layout()
plt.show()

# --- CNEES and CNIS Calculation ---
# Now just plot the CNEES and CNIS columns from the CSV as separate figures

plt.figure(figsize=(10, 4))
plt.plot(data['cnees'].values, label='CNEES')
plt.xlabel('Time step')
plt.ylabel('CNEES Value')
plt.title('CNEES over Time')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(data['cnis'].values, label='CNIS', color='orange')
plt.xlabel('Time step')
plt.ylabel('CNIS Value')
plt.title('CNIS over Time')
plt.legend()
plt.tight_layout()
plt.show()

# NOTE: Replace the identity matrices with the actual state and innovation covariances for correct results. 
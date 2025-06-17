import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

csv_path = "/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/build/2d_progress.csv"

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# True position (hardcoded to match C++ code)
true_position = np.array([5.0, 3.0])

while True:
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        
        # Convert to numpy arrays
        iterations = np.array(data['iteration'])
        x_positions = np.array(data['x'])
        y_positions = np.array(data['y'])
        errors = np.array(data['total_error'])
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot trajectory
        ax1.plot(x_positions, y_positions, 'b-o', label='Estimated Position')
        ax1.plot(true_position[0], true_position[1], 'r*', markersize=15, label='True Position')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('2D Position Estimation')
        ax1.grid(True)
        ax1.legend()
        
        # Plot error
        ax2.plot(iterations, errors, 'r-', label='Total Error')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Total Error')
        ax2.set_title('Optimization Error')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.pause(0.5)
    else:
        print("Waiting for 2d_progress.csv...")
        time.sleep(1) 
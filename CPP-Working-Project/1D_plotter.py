import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

csv_path = "/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/build/progress.csv"

plt.ion()
fig, ax1 = plt.subplots()

while True:
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        # Convert to numpy arrays
        iterations = np.array(data['iteration'])
        estimates = np.array(data['estimate'])
        errors = np.array(data['total_error'])
        
        ax1.clear()
        # Plot estimate
        ax1.plot(iterations, estimates, 'b-o', label='Estimate')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Estimate', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot error on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(iterations, errors, 'r-x', label='Total Error')
        ax2.set_ylabel('Total Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('1D Factor Graph Optimization Progress')
        plt.tight_layout()
        plt.pause(0.5)
    else:
        print("Waiting for progress.csv...")
        time.sleep(1)
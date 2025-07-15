import numpy as np
import matplotlib.pyplot as plt
import h5py

# Ask user for 1D or 2D
choice = input('Show chi2 results for (1) 1D or (2) 2D? Enter 1 or 2: ').strip()
if choice == '1':
    h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/1D/1D_chi2_results.h5'
    title = 'Chi2 Values Across Runs (1D)'
else:
    h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2D/2D_chi2_results.h5'
    title = 'Chi2 Values Across Runs (2D)'

# Read the HDF5 file
with h5py.File(h5_path, 'r') as f:
    chi2_values = f['chi2'][:]
runs = np.arange(len(chi2_values))

# Calculate and print the average chi2 value
average_chi2 = np.mean(chi2_values)
print(f'Average Chi2 Value: {average_chi2}') 

# Plot chi2 values
plt.figure(figsize=(10, 6))
plt.plot(runs, chi2_values, marker='o', linestyle='-', markersize=2)
plt.axhline(average_chi2, color='red', linestyle='--', label=f'Average = {average_chi2:.2f}')
plt.xlabel('Run')
plt.ylabel('Chi2 Value')
plt.title(title)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

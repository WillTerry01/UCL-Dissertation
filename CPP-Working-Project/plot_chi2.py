import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skipping the header
chi2_data = np.genfromtxt('chi2_results.csv', delimiter=',', skip_header=1)
runs = chi2_data[:, 0]
chi2_values = chi2_data[:, 1]

# Calculate and print the average chi2 value
average_chi2 = np.mean(chi2_values)
print(f'Average Chi2 Value: {average_chi2}') 

# Plot chi2 values
plt.figure(figsize=(10, 6))
plt.plot(runs, chi2_values, marker='o', linestyle='-', markersize=2)
plt.axhline(average_chi2, color='red', linestyle='--', label=f'Average = {average_chi2:.2f}')
plt.xlabel('Run')
plt.ylabel('Chi2 Value')
plt.title('Chi2 Values Across Runs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

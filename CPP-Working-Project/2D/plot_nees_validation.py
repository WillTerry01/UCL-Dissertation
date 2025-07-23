#!/usr/bin/env python3
"""
Plot NEES validation results from HDF5 file.
Plots mean and variance of NEES values over time steps.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_nees_data(filename):
    """Load NEES data from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        # Load datasets
        monte_carlo_runs = f['monte_carlo_runs'][:]
        nees_means = f['nees_means'][:]
        nees_variances = f['nees_variances'][:]
        
        # Load attributes
        overall_mean = f.attrs['overall_mean']
        overall_variance = f.attrs['overall_variance']
        theoretical_mean = f.attrs['theoretical_mean']
        theoretical_variance = f.attrs['theoretical_variance']
        Q = f.attrs['Q']
        R = f.attrs['R']
        
    return {
        'monte_carlo_runs': monte_carlo_runs,
        'nees_means': nees_means,
        'nees_variances': nees_variances,
        'overall_mean': overall_mean,
        'overall_variance': overall_variance,
        'theoretical_mean': theoretical_mean,
        'theoretical_variance': theoretical_variance,
        'Q': Q,
        'R': R
    }

def plot_nees_validation(data, save_plot=True):
    """Create comprehensive NEES validation plots."""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'NEES Validation Results (Q={data["Q"]:.6f}, R={data["R"]:.6f})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: NEES Mean over Monte Carlo runs
    ax1.plot(data['monte_carlo_runs'], data['nees_means'], 'b-', linewidth=2, label='Calculated Mean')
    ax1.axhline(y=data['theoretical_mean'], color='r', linestyle='--', linewidth=2, 
                label=f'Theoretical Mean ({data["theoretical_mean"]:.1f})')
    ax1.axhline(y=data['overall_mean'], color='g', linestyle=':', linewidth=2,
                label=f'Overall Mean ({data["overall_mean"]:.3f})')
    
    ax1.set_xlabel('Monte Carlo Run', fontsize=12)
    ax1.set_ylabel('NEES Mean', fontsize=12)
    ax1.set_title('NEES Mean vs Monte Carlo Run', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Overall Mean: {data["overall_mean"]:.4f}\nTheoretical: {data["theoretical_mean"]:.1f}\nDifference: {abs(data["overall_mean"] - data["theoretical_mean"]):.4f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: NEES Variance over Monte Carlo runs
    ax2.plot(data['monte_carlo_runs'], data['nees_variances'], 'purple', linewidth=2, label='Calculated Variance')
    ax2.axhline(y=data['theoretical_variance'], color='r', linestyle='--', linewidth=2,
                label=f'Theoretical Variance ({data["theoretical_variance"]:.1f})')
    ax2.axhline(y=data['overall_variance'], color='g', linestyle=':', linewidth=2,
                label=f'Overall Variance ({data["overall_variance"]:.3f})')
    
    ax2.set_xlabel('Monte Carlo Run', fontsize=12)
    ax2.set_ylabel('NEES Variance', fontsize=12)
    ax2.set_title('NEES Variance vs Monte Carlo Run', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Overall Variance: {data["overall_variance"]:.4f}\nTheoretical: {data["theoretical_variance"]:.1f}\nRatio: {data["overall_variance"]/data["theoretical_variance"]:.2f}x'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('../H5_Files/2D_nees_validation_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to ../H5_Files/2D_nees_validation_plot.png")
    
    plt.show()

def plot_combined_view(data, save_plot=True):
    """Create a combined view showing both mean and variance on same plot."""
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot mean on primary y-axis
    line1 = ax1.plot(data['monte_carlo_runs'], data['nees_means'], 'b-', linewidth=2, label='NEES Mean')
    ax1.axhline(y=data['theoretical_mean'], color='b', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Theoretical Mean ({data["theoretical_mean"]:.1f})')
    
    # Plot variance on secondary y-axis
    line2 = ax2.plot(data['monte_carlo_runs'], data['nees_variances'], 'r-', linewidth=2, label='NEES Variance')
    ax2.axhline(y=data['theoretical_variance'], color='r', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Theoretical Variance ({data["theoretical_variance"]:.1f})')
    
    # Set labels and title
    ax1.set_xlabel('Monte Carlo Run', fontsize=12)
    ax1.set_ylabel('NEES Mean', color='b', fontsize=12)
    ax2.set_ylabel('NEES Variance', color='r', fontsize=12)
    ax1.set_title(f'NEES Mean and Variance vs Monte Carlo Run\n(Q={data["Q"]:.6f}, R={data["R"]:.6f})', 
                  fontsize=14, fontweight='bold')
    
    # Set colors for y-axis labels
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Add statistics text box
    stats_text = f'Overall Mean: {data["overall_mean"]:.4f}\nOverall Variance: {data["overall_variance"]:.4f}\nVariance Ratio: {data["overall_variance"]/data["theoretical_variance"]:.2f}x'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('../H5_Files/2D_nees_validation_combined.png', dpi=300, bbox_inches='tight')
        print("Combined plot saved to ../H5_Files/2D_nees_validation_combined.png")
    
    plt.show()

def analyze_run_distribution(data):
    """Analyze and plot distribution of NEES values across Monte Carlo runs."""
    
    # Calculate statistics across runs
    mean_of_means = np.mean(data['nees_means'])
    std_of_means = np.std(data['nees_means'])
    mean_of_variances = np.mean(data['nees_variances'])
    std_of_variances = np.std(data['nees_variances'])
    
    # Create distribution analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NEES Distribution Analysis Across Monte Carlo Runs', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram of means
    ax1.hist(data['nees_means'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(mean_of_means, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_of_means:.3f}')
    ax1.axvline(data['theoretical_mean'], color='green', linestyle='--', linewidth=2, label=f'Theoretical: {data["theoretical_mean"]:.1f}')
    ax1.set_xlabel('NEES Mean per Run', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of NEES Means', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of variances
    ax2.hist(data['nees_variances'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(mean_of_variances, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_of_variances:.3f}')
    ax2.axvline(data['theoretical_variance'], color='green', linestyle='--', linewidth=2, label=f'Theoretical: {data["theoretical_variance"]:.1f}')
    ax2.set_xlabel('NEES Variance per Run', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of NEES Variances', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of mean vs variance
    ax3.scatter(data['nees_means'], data['nees_variances'], alpha=0.6, color='orange')
    ax3.axhline(y=data['theoretical_variance'], color='green', linestyle='--', alpha=0.7, label=f'Theoretical Variance: {data["theoretical_variance"]:.1f}')
    ax3.axvline(x=data['theoretical_mean'], color='green', linestyle='--', alpha=0.7, label=f'Theoretical Mean: {data["theoretical_mean"]:.1f}')
    ax3.set_xlabel('NEES Mean per Run', fontsize=12)
    ax3.set_ylabel('NEES Variance per Run', fontsize=12)
    ax3.set_title('Mean vs Variance Correlation', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot
    ax4.boxplot([data['nees_means'], data['nees_variances'], 
                 [data['theoretical_mean']] * len(data['nees_means']), 
                 [data['theoretical_variance']] * len(data['nees_variances'])], 
                labels=['NEES Means', 'NEES Variances', 'Theoretical Mean', 'Theoretical Variance'])
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Box Plot Comparison', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../H5_Files/2D_nees_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("Distribution analysis plot saved to ../H5_Files/2D_nees_distribution_analysis.png")
    plt.show()
    
    # Print statistics
    print(f"\nDistribution Analysis:")
    print(f"NEES Means: {mean_of_means:.4f} ± {std_of_means:.4f}")
    print(f"NEES Variances: {mean_of_variances:.4f} ± {std_of_variances:.4f}")
    print(f"Mean/Theory Ratio: {mean_of_means/data['theoretical_mean']:.3f}")
    print(f"Variance/Theory Ratio: {mean_of_variances/data['theoretical_variance']:.3f}")
    print(f"Correlation (Mean vs Variance): {np.corrcoef(data['nees_means'], data['nees_variances'])[0,1]:.3f}")

def main():
    """Main function to load data and create plots."""
    
    filename = "../H5_Files/2D_nees_validation_results.h5"
    
    try:
        print(f"Loading NEES data from {filename}...")
        data = load_nees_data(filename)
        
        print(f"Data loaded successfully!")
        print(f"Parameters: Q={data['Q']:.6f}, R={data['R']:.6f}")
        print(f"Overall Mean: {data['overall_mean']:.4f} (Theoretical: {data['theoretical_mean']:.1f})")
        print(f"Overall Variance: {data['overall_variance']:.4f} (Theoretical: {data['theoretical_variance']:.1f})")
        print(f"Variance Ratio: {data['overall_variance']/data['theoretical_variance']:.2f}x")
        
        # Create plots
        print("\nCreating plots...")
        plot_nees_validation(data)
        plot_combined_view(data)
        analyze_run_distribution(data)
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Please run the validation filter first to generate the HDF5 file.")
    except Exception as e:
        print(f"Error loading or plotting data: {e}")

if __name__ == "__main__":
    main() 
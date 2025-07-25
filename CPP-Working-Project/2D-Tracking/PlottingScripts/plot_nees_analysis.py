#!/usr/bin/env python3
"""
NEES Analysis Visualization Script
Plots NEES means and variances from validation results with cumulative statistics.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os

def load_nees_results(filename):
    """Load NEES validation results from HDF5 file."""
    print(f"Loading NEES results from: {filename}")
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print("Please run './Validate_QR' first to generate the validation data.")
        return None
    
    try:
        with h5py.File(filename, 'r') as f:
            # Load datasets
            monte_carlo_runs = np.array(f['monte_carlo_runs'])
            nees_values = np.array(f['nees_full_system_values'])
            
            # Load metadata
            overall_mean = f.attrs['overall_mean']
            overall_variance = f.attrs['overall_variance']
            theoretical_mean = f.attrs['theoretical_mean']
            theoretical_variance = f.attrs['theoretical_variance']
            V0 = f.attrs['V0']
            meas_noise_std = f.attrs['meas_noise_std']
            total_dof = f.attrs['total_degrees_of_freedom']
            use_existing_data = bool(f.attrs['use_existing_data'])
            method = f.attrs['method'].decode('utf-8') if 'method' in f.attrs else 'unknown'
            
            print(f"Loaded {len(nees_values)} NEES values")
            print(f"Parameters: V0={V0:.4f}, meas_noise_std={meas_noise_std:.4f}")
            print(f"Method: {method}, Total DOF: {total_dof}")
            print(f"Data source: {'Existing data' if use_existing_data else 'New data'}")
            
            return {
                'runs': monte_carlo_runs,
                'nees_values': nees_values,
                'overall_mean': overall_mean,
                'overall_variance': overall_variance,
                'theoretical_mean': theoretical_mean,
                'theoretical_variance': theoretical_variance,
                'V0': V0,
                'meas_noise_std': meas_noise_std,
                'total_dof': total_dof,
                'use_existing_data': use_existing_data,
                'method': method
            }
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def calculate_cumulative_statistics(nees_values):
    """Calculate cumulative mean and variance as a function of number of runs."""
    n_runs = len(nees_values)
    cumulative_means = np.zeros(n_runs)
    cumulative_variances = np.zeros(n_runs)
    
    for i in range(1, n_runs + 1):
        subset = nees_values[:i]
        cumulative_means[i-1] = np.mean(subset)
        if i > 1:
            cumulative_variances[i-1] = np.var(subset, ddof=1)  # Sample variance
        else:
            cumulative_variances[i-1] = 0  # Undefined for single sample
    
    return cumulative_means, cumulative_variances

def plot_nees_convergence(data):
    """Plot NEES mean and variance convergence with number of runs."""
    runs = data['runs']
    nees_values = data['nees_values']
    
    # Calculate cumulative statistics
    cum_means, cum_variances = calculate_cumulative_statistics(nees_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Combined NEES Mean and Variance Convergence
    ax1 = axes[0, 0]
    
    # Plot cumulative mean and variance on same axes
    line1 = ax1.plot(runs, cum_means, 'b-', linewidth=3, 
                     label=f'Cumulative Mean (avg: {data["overall_mean"]:.2f})')
    line2 = ax1.plot(runs, cum_variances, 'r-', linewidth=3, 
                     label=f'Cumulative Variance (avg: {data["overall_variance"]:.2f})')
    
    # Add theoretical reference lines
    ax1.axhline(y=data['theoretical_mean'], color='blue', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    ax1.axhline(y=data['theoretical_variance'], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Theoretical Variance = {data["theoretical_variance"]:.0f}')
    
    # Set y-axis range from 0 to max variance + 10% margin
    max_value = max(np.max(cum_variances), data['theoretical_variance'])
    y_margin = max_value * 0.1
    ax1.set_ylim(0, max_value + y_margin)
    
    ax1.set_xlabel('Number of Monte Carlo Runs')
    ax1.set_ylabel('NEES Value')
    ax1.set_title('NEES Mean and Variance Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view of Mean Convergence with confidence bands
    ax2 = axes[0, 1]
    ax2.plot(runs, cum_means, 'b-', linewidth=2, label='Cumulative Mean')
    ax2.axhline(y=data['theoretical_mean'], color='red', linestyle='--', 
                linewidth=2, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    ax2.axhline(y=data['overall_mean'], color='green', linestyle=':', 
                linewidth=2, label=f'Final Mean = {data["overall_mean"]:.2f}')
    
    # Add confidence bands (±2σ/√n for mean)
    std_error = np.sqrt(cum_variances / runs)
    ax2.fill_between(runs[1:], (cum_means - 2*std_error)[1:], (cum_means + 2*std_error)[1:], 
                     alpha=0.3, color='blue', label='±2σ/√n')
    
    ax2.set_xlabel('Number of Monte Carlo Runs')
    ax2.set_ylabel('NEES Mean')
    ax2.set_title('NEES Mean Convergence (Detailed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual NEES values
    ax3 = axes[1, 0]
    ax3.scatter(runs, nees_values, alpha=0.6, s=20, c='purple', label='Individual NEES')
    ax3.axhline(y=data['theoretical_mean'], color='blue', linestyle='--', 
                linewidth=2, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    ax3.axhline(y=data['theoretical_variance'], color='red', linestyle='--', 
                linewidth=2, label=f'Theoretical Variance = {data["theoretical_variance"]:.0f}')
    
    # Add theoretical confidence intervals (assuming chi-squared distribution)
    # For large DOF, NEES ~ N(n, 2n)
    theoretical_std = np.sqrt(data['theoretical_variance'])
    ax3.axhspan(data['theoretical_mean'] - 2*theoretical_std, 
                data['theoretical_mean'] + 2*theoretical_std,
                alpha=0.2, color='blue', label='Mean ±2σ')
    
    # Use same y-axis scaling as main plot
    ax3.set_ylim(0, max_value + y_margin)
    
    ax3.set_xlabel('Monte Carlo Run Number')
    ax3.set_ylabel('NEES Value')
    ax3.set_title('Individual NEES Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative error convergence
    ax4 = axes[1, 1]
    relative_mean_error = np.abs(cum_means - data['theoretical_mean']) / data['theoretical_mean'] * 100
    relative_var_error = np.abs(cum_variances - data['theoretical_variance']) / data['theoretical_variance'] * 100
    
    ax4.plot(runs, relative_mean_error, 'b-', linewidth=2, 
             label=f'Mean Error (final: {relative_mean_error[-1]:.2f}%)')
    ax4.plot(runs[1:], relative_var_error[1:], 'r-', linewidth=2, 
             label=f'Variance Error (final: {relative_var_error[-1]:.2f}%)')
    
    ax4.set_xlabel('Number of Monte Carlo Runs')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Convergence of Relative Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle(f'NEES Convergence Analysis\n' +
                 f'V0={data["V0"]:.4f}, σ_meas={data["meas_noise_std"]:.4f}, DOF={data["total_dof"]}', 
                 fontsize=14)
    plt.tight_layout()
    return fig

def plot_nees_distribution(data):
    """Plot NEES distribution analysis."""
    nees_values = data['nees_values']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Histogram of NEES values
    ax1 = axes[0]
    n_bins = min(50, int(np.sqrt(len(nees_values))))
    counts, bins, patches = ax1.hist(nees_values, bins=n_bins, alpha=0.7, color='skyblue', 
                                     density=True, edgecolor='navy')
    
    # Overlay theoretical chi-squared distribution (approximated as normal for large DOF)
    if data['total_dof'] > 30:  # Normal approximation valid for large DOF
        x_theory = np.linspace(nees_values.min(), nees_values.max(), 200)
        theoretical_pdf = (1/np.sqrt(2*np.pi*data['theoretical_variance'])) * \
                          np.exp(-0.5*(x_theory - data['theoretical_mean'])**2 / data['theoretical_variance'])
        ax1.plot(x_theory, theoretical_pdf, 'r-', linewidth=2, label='Theoretical (Normal approx.)')
    
    ax1.axvline(x=data['overall_mean'], color='green', linestyle='--', 
                linewidth=2, label=f'Sample Mean = {data["overall_mean"]:.2f}')
    ax1.axvline(x=data['theoretical_mean'], color='red', linestyle=':', 
                linewidth=2, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    
    ax1.set_xlabel('NEES Value')
    ax1.set_ylabel('Density')
    ax1.set_title('NEES Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot against theoretical distribution
    ax2 = axes[1]
    sorted_nees = np.sort(nees_values)
    n = len(sorted_nees)
    theoretical_quantiles = np.linspace(0.01, 0.99, n)
    
    if data['total_dof'] > 30:  # Normal approximation
        from scipy.stats import norm
        theoretical_values = norm.ppf(theoretical_quantiles, 
                                      loc=data['theoretical_mean'], 
                                      scale=np.sqrt(data['theoretical_variance']))
        ax2.scatter(theoretical_values, sorted_nees, alpha=0.6, s=20, color='blue')
        
        # Perfect fit line
        min_val = min(theoretical_values.min(), sorted_nees.min())
        max_val = max(theoretical_values.max(), sorted_nees.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plot (vs Normal)')
    else:
        # For small DOF, use chi-squared directly
        from scipy.stats import chi2
        theoretical_values = chi2.ppf(theoretical_quantiles, df=data['total_dof'])
        ax2.scatter(theoretical_values, sorted_nees, alpha=0.6, s=20, color='blue')
        
        min_val = min(theoretical_values.min(), sorted_nees.min())
        max_val = max(theoretical_values.max(), sorted_nees.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title(f'Q-Q Plot (vs χ²({data["total_dof"]}))')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Running statistics
    ax3 = axes[2]
    window_size = max(10, len(nees_values) // 20)  # Adaptive window size
    running_means = []
    running_vars = []
    window_centers = []
    
    for i in range(window_size, len(nees_values) + 1):
        window = nees_values[i-window_size:i]
        running_means.append(np.mean(window))
        running_vars.append(np.var(window, ddof=1))
        window_centers.append(i - window_size/2)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(window_centers, running_means, 'b-', linewidth=2, label='Running Mean')
    ax3.axhline(y=data['theoretical_mean'], color='blue', linestyle='--', alpha=0.7)
    
    line2 = ax3_twin.plot(window_centers, running_vars, 'r-', linewidth=2, label='Running Variance')
    ax3_twin.axhline(y=data['theoretical_variance'], color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel(f'Run Number (Window size: {window_size})')
    ax3.set_ylabel('NEES Mean', color='blue')
    ax3_twin.set_ylabel('NEES Variance', color='red')
    ax3.set_title('Running Statistics')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_parameter_summary(data):
    """Create a summary plot with key statistics."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a text summary
    summary_text = f"""NEES Validation Summary
    
Parameters:
• Process noise intensity (V0): {data['V0']:.4f}
• Measurement noise std (σ): {data['meas_noise_std']:.4f}
• Total degrees of freedom: {data['total_dof']}
• Method: {data['method']}
• Data source: {'Existing data' if data['use_existing_data'] else 'New data'}

Results:
• Number of Monte Carlo runs: {len(data['nees_values'])}
• Sample mean: {data['overall_mean']:.4f}
• Theoretical mean: {data['theoretical_mean']:.4f}
• Mean error: {abs(data['overall_mean'] - data['theoretical_mean']):.4f} ({abs(data['overall_mean'] - data['theoretical_mean'])/data['theoretical_mean']*100:.2f}%)

• Sample variance: {data['overall_variance']:.4f}
• Theoretical variance: {data['theoretical_variance']:.4f}
• Variance error: {abs(data['overall_variance'] - data['theoretical_variance']):.4f} ({abs(data['overall_variance'] - data['theoretical_variance'])/data['theoretical_variance']*100:.2f}%)

Quality Assessment:
• Mean within 5% of theory: {'✓' if abs(data['overall_mean'] - data['theoretical_mean'])/data['theoretical_mean'] < 0.05 else '✗'}
• Variance within 20% of theory: {'✓' if abs(data['overall_variance'] - data['theoretical_variance'])/data['theoretical_variance'] < 0.20 else '✗'}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('NEES Validation Summary', fontsize=16, fontweight='bold')
    
    return fig

def main():
    """Main function to generate all NEES analysis plots."""
    print("=== NEES Analysis Visualization ===")
    
    # Try to load both validation files
    files_to_check = [
        ("../2D-Tracking/Saved_Data/2D_nees_validation_same_data.h5", "Same Data"),
        ("../2D-Tracking/Saved_Data/2D_nees_validation_new_data.h5", "New Data")
    ]
    
    os.makedirs("../2D-Tracking/plots", exist_ok=True)
    
    for data_file, data_type in files_to_check:
        print(f"\n--- Processing {data_type} ---")
        data = load_nees_results(data_file)
        
        if data is None:
            print(f"Skipping {data_type} - file not found")
            continue
        
        # Generate plots
        print(f"Creating convergence plot for {data_type}...")
        fig1 = plot_nees_convergence(data)
        fig1.savefig(f"../2D-Tracking/plots/nees_convergence_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print(f"Creating distribution plot for {data_type}...")
        fig2 = plot_nees_distribution(data)
        fig2.savefig(f"../2D-Tracking/plots/nees_distribution_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"Creating summary for {data_type}...")
        fig3 = plot_parameter_summary(data)
        fig3.savefig(f"../2D-Tracking/plots/nees_summary_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    print("\n=== NEES Analysis Complete ===")
    print("Generated plots saved in '../2D-Tracking/plots/' directory:")
    print("  - nees_convergence_*.png: Mean and variance convergence with number of runs")
    print("  - nees_distribution_*.png: Distribution analysis and Q-Q plots")
    print("  - nees_summary_*.png: Summary of validation results")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
NIS Analysis Visualization Script
Plots NIS means and variances from validation results with cumulative statistics.
Supports both Khosoussi Proposition 3 and 4 based on YAML configuration.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import yaml

def load_yaml_config():
    """Load configuration from YAML file to determine NIS method."""
    config_file = "../BO_Parameters.yaml"
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        consistency_method = config.get('bayesopt', {}).get('consistency_method', 'nees')
        
        if consistency_method in ['nis3', 'nis4']:
            print(f"Detected consistency method: {consistency_method}")
            return consistency_method
        else:
            print(f"Warning: Consistency method '{consistency_method}' is not NIS-based")
            return None
            
    except Exception as e:
        print(f"Error reading YAML config: {e}")
        return None

def calculate_dof_for_method(consistency_method, trajectory_length=50):
    """Calculate degrees of freedom based on NIS method."""
    nx = 4  # state dimension
    nz = 2  # measurement dimension
    T = trajectory_length
    
    if consistency_method == "nis3":
        # Proposition 3: DOF = Nz (total residuals)
        Nz = T * nz + (T - 1) * nx
        return Nz
    elif consistency_method == "nis4":
        # Proposition 4: DOF = Nz - Nx (total residuals - total state parameters)
        Nx = T * nx
        Nz = T * nz + (T - 1) * nx
        return Nz - Nx
    else:
        raise ValueError(f"Unknown consistency method: {consistency_method}")

def load_nis_results(filename, consistency_method):
    """Load NIS validation results from HDF5 file."""
    print(f"Loading {consistency_method.upper()} results from: {filename}")
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print("Please run './Validate_QR' with the appropriate consistency method first.")
        return None
    
    try:
        with h5py.File(filename, 'r') as f:
            # Load datasets - the dataset name depends on the consistency method
            monte_carlo_runs = np.array(f['monte_carlo_runs'])
            dataset_name = f"{consistency_method}_full_system_values"
            
            if dataset_name not in f:
                print(f"Error: Dataset '{dataset_name}' not found in file")
                print(f"Available datasets: {list(f.keys())}")
                return None
                
            nis_values = np.array(f[dataset_name])
            
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
            
            print(f"Loaded {len(nis_values)} {consistency_method.upper()} values")
            print(f"Parameters: V0={V0:.4f}, meas_noise_std={meas_noise_std:.4f}")
            print(f"Method: {method}, Total DOF: {total_dof}")
            print(f"Data source: {'Existing data' if use_existing_data else 'New data'}")
            
            return {
                'runs': monte_carlo_runs,
                'nis_values': nis_values,
                'overall_mean': overall_mean,
                'overall_variance': overall_variance,
                'theoretical_mean': theoretical_mean,
                'theoretical_variance': theoretical_variance,
                'V0': V0,
                'meas_noise_std': meas_noise_std,
                'total_dof': total_dof,
                'use_existing_data': use_existing_data,
                'method': method,
                'consistency_method': consistency_method
            }
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def calculate_cumulative_statistics(nis_values):
    """Calculate cumulative mean and variance as a function of number of runs."""
    n_runs = len(nis_values)
    cumulative_means = np.zeros(n_runs)
    cumulative_variances = np.zeros(n_runs)
    
    for i in range(1, n_runs + 1):
        subset = nis_values[:i]
        cumulative_means[i-1] = np.mean(subset)
        if i > 1:
            cumulative_variances[i-1] = np.var(subset, ddof=1)  # Sample variance
        else:
            cumulative_variances[i-1] = 0  # Undefined for single sample
    
    return cumulative_means, cumulative_variances

def plot_nis_convergence(data):
    """Plot NIS mean and variance convergence with number of runs."""
    runs = data['runs']
    nis_values = data['nis_values']
    consistency_method = data['consistency_method'].upper()
    
    # Calculate cumulative statistics
    cum_means, cum_variances = calculate_cumulative_statistics(nis_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Combined NIS Mean and Variance Convergence
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
    ax1.set_ylabel(f'{consistency_method} Value')
    ax1.set_title(f'{consistency_method} Mean and Variance Convergence')
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
    ax2.set_ylabel(f'{consistency_method} Mean')
    ax2.set_title(f'{consistency_method} Mean Convergence (Detailed)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual NIS values
    ax3 = axes[1, 0]
    ax3.scatter(runs, nis_values, alpha=0.6, s=20, c='purple', label=f'Individual {consistency_method}')
    ax3.axhline(y=data['theoretical_mean'], color='blue', linestyle='--', 
                linewidth=2, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    ax3.axhline(y=data['theoretical_variance'], color='red', linestyle='--', 
                linewidth=2, label=f'Theoretical Variance = {data["theoretical_variance"]:.0f}')
    
    # Add theoretical confidence intervals (assuming chi-squared distribution)
    # For large DOF, NIS ~ N(n, 2n)
    theoretical_std = np.sqrt(data['theoretical_variance'])
    ax3.axhspan(data['theoretical_mean'] - 2*theoretical_std, 
                data['theoretical_mean'] + 2*theoretical_std,
                alpha=0.2, color='blue', label='Mean ±2σ')
    
    # Use same y-axis scaling as main plot
    ax3.set_ylim(0, max_value + y_margin)
    
    ax3.set_xlabel('Monte Carlo Run Number')
    ax3.set_ylabel(f'{consistency_method} Value')
    ax3.set_title(f'Individual {consistency_method} Values')
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
    
    method_desc = f"Khosoussi Proposition {'3' if 'nis3' in data['consistency_method'] else '4'}"
    plt.suptitle(f'{consistency_method} Convergence Analysis ({method_desc})\n' +
                 f'V0={data["V0"]:.4f}, σ_meas={data["meas_noise_std"]:.4f}, DOF={data["total_dof"]}', 
                 fontsize=14)
    plt.tight_layout()
    return fig

def plot_nis_distribution(data):
    """Plot NIS distribution analysis."""
    nis_values = data['nis_values']
    consistency_method = data['consistency_method'].upper()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Histogram of NIS values
    ax1 = axes[0]
    n_bins = min(50, int(np.sqrt(len(nis_values))))
    counts, bins, patches = ax1.hist(nis_values, bins=n_bins, alpha=0.7, color='lightcoral', 
                                     density=True, edgecolor='darkred')
    
    # Overlay theoretical chi-squared distribution (approximated as normal for large DOF)
    if data['total_dof'] > 30:  # Normal approximation valid for large DOF
        x_theory = np.linspace(nis_values.min(), nis_values.max(), 200)
        theoretical_pdf = (1/np.sqrt(2*np.pi*data['theoretical_variance'])) * \
                          np.exp(-0.5*(x_theory - data['theoretical_mean'])**2 / data['theoretical_variance'])
        ax1.plot(x_theory, theoretical_pdf, 'r-', linewidth=2, label='Theoretical (Normal approx.)')
    
    ax1.axvline(x=data['overall_mean'], color='green', linestyle='--', 
                linewidth=2, label=f'Sample Mean = {data["overall_mean"]:.2f}')
    ax1.axvline(x=data['theoretical_mean'], color='red', linestyle=':', 
                linewidth=2, label=f'Theoretical Mean = {data["theoretical_mean"]:.0f}')
    
    ax1.set_xlabel(f'{consistency_method} Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{consistency_method} Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot against theoretical distribution
    ax2 = axes[1]
    sorted_nis = np.sort(nis_values)
    n = len(sorted_nis)
    theoretical_quantiles = np.linspace(0.01, 0.99, n)
    
    if data['total_dof'] > 30:  # Normal approximation
        from scipy.stats import norm
        theoretical_values = norm.ppf(theoretical_quantiles, 
                                      loc=data['theoretical_mean'], 
                                      scale=np.sqrt(data['theoretical_variance']))
        ax2.scatter(theoretical_values, sorted_nis, alpha=0.6, s=20, color='blue')
        
        # Perfect fit line
        min_val = min(theoretical_values.min(), sorted_nis.min())
        max_val = max(theoretical_values.max(), sorted_nis.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plot (vs Normal)')
    else:
        # For small DOF, use chi-squared directly
        from scipy.stats import chi2
        theoretical_values = chi2.ppf(theoretical_quantiles, df=data['total_dof'])
        ax2.scatter(theoretical_values, sorted_nis, alpha=0.6, s=20, color='blue')
        
        min_val = min(theoretical_values.min(), sorted_nis.min())
        max_val = max(theoretical_values.max(), sorted_nis.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title(f'Q-Q Plot (vs χ²({data["total_dof"]}))')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Running statistics
    ax3 = axes[2]
    window_size = max(10, len(nis_values) // 20)  # Adaptive window size
    running_means = []
    running_vars = []
    window_centers = []
    
    for i in range(window_size, len(nis_values) + 1):
        window = nis_values[i-window_size:i]
        running_means.append(np.mean(window))
        running_vars.append(np.var(window, ddof=1))
        window_centers.append(i - window_size/2)
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(window_centers, running_means, 'b-', linewidth=2, label='Running Mean')
    ax3.axhline(y=data['theoretical_mean'], color='blue', linestyle='--', alpha=0.7)
    
    line2 = ax3_twin.plot(window_centers, running_vars, 'r-', linewidth=2, label='Running Variance')
    ax3_twin.axhline(y=data['theoretical_variance'], color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel(f'Run Number (Window size: {window_size})')
    ax3.set_ylabel(f'{consistency_method} Mean', color='blue')
    ax3_twin.set_ylabel(f'{consistency_method} Variance', color='red')
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
    consistency_method = data['consistency_method'].upper()
    
    # Determine proposition description
    if 'nis3' in data['consistency_method']:
        prop_desc = "Khosoussi Proposition 3 (DOF = Nz)"
    else:
        prop_desc = "Khosoussi Proposition 4 (DOF = Nz - Nx)"
    
    # Create a text summary
    summary_text = f"""{consistency_method} Validation Summary

Method: {prop_desc}
    
Parameters:
• Process noise intensity (V0): {data['V0']:.4f}
• Measurement noise std (σ): {data['meas_noise_std']:.4f}
• Total degrees of freedom: {data['total_dof']}
• Implementation: {data['method']}
• Data source: {'Existing data' if data['use_existing_data'] else 'New data'}

Results:
• Number of Monte Carlo runs: {len(data['nis_values'])}
• Sample mean: {data['overall_mean']:.4f}
• Theoretical mean: {data['theoretical_mean']:.4f}
• Mean error: {abs(data['overall_mean'] - data['theoretical_mean']):.4f} ({abs(data['overall_mean'] - data['theoretical_mean'])/data['theoretical_mean']*100:.2f}%)

• Sample variance: {data['overall_variance']:.4f}
• Theoretical variance: {data['theoretical_variance']:.4f}
• Variance error: {abs(data['overall_variance'] - data['theoretical_variance']):.4f} ({abs(data['overall_variance'] - data['theoretical_variance'])/data['theoretical_variance']*100:.2f}%)

Quality Assessment:
• Mean within 5% of theory: {'✓' if abs(data['overall_mean'] - data['theoretical_mean'])/data['theoretical_mean'] < 0.05 else '✗'}
• Variance within 20% of theory: {'✓' if abs(data['overall_variance'] - data['theoretical_variance'])/data['theoretical_variance'] < 0.20 else '✗'}

NIS vs NEES:
• NIS tests measurement prediction quality (innovation space)
• NEES tests state estimation quality (state space)
• Different DOF calculations for different theoretical foundations
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'{consistency_method} Validation Summary ({prop_desc})', 
                 fontsize=16, fontweight='bold')
    
    return fig

def main():
    """Main function to generate all NIS analysis plots."""
    print("=== NIS Analysis Visualization ===")
    
    # Load YAML configuration to determine NIS method
    consistency_method = load_yaml_config()
    
    if consistency_method is None:
        print("Error: Could not determine NIS method from YAML configuration.")
        print("Please ensure bayesopt.consistency_method is set to 'nis3' or 'nis4'")
        return
    
    if consistency_method not in ['nis3', 'nis4']:
        print(f"Error: Consistency method '{consistency_method}' is not NIS-based.")
        print("This script only works with 'nis3' or 'nis4' methods.")
        return
    
    # Try to load both validation files
    files_to_check = [
        (f"../2D-Tracking/Saved_Data/2D_{consistency_method}_validation_same_data.h5", "Same Data"),
        (f"../2D-Tracking/Saved_Data/2D_{consistency_method}_validation_new_data.h5", "New Data")
    ]
    
    os.makedirs("../2D-Tracking/plots", exist_ok=True)
    
    for data_file, data_type in files_to_check:
        print(f"\n--- Processing {data_type} for {consistency_method.upper()} ---")
        data = load_nis_results(data_file, consistency_method)
        
        if data is None:
            print(f"Skipping {data_type} - file not found")
            continue
        
        # Generate plots
        print(f"Creating convergence plot for {data_type}...")
        fig1 = plot_nis_convergence(data)
        fig1.savefig(f"../2D-Tracking/plots/{consistency_method}_convergence_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print(f"Creating distribution plot for {data_type}...")
        fig2 = plot_nis_distribution(data)
        fig2.savefig(f"../2D-Tracking/plots/{consistency_method}_distribution_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"Creating summary for {data_type}...")
        fig3 = plot_parameter_summary(data)
        fig3.savefig(f"../2D-Tracking/plots/{consistency_method}_summary_{data_type.lower().replace(' ', '_')}.png", 
                     dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    print(f"\n=== {consistency_method.upper()} Analysis Complete ===")
    print("Generated plots saved in '../2D-Tracking/plots/' directory:")
    print(f"  - {consistency_method}_convergence_*.png: Mean and variance convergence with number of runs")
    print(f"  - {consistency_method}_distribution_*.png: Distribution analysis and Q-Q plots")
    print(f"  - {consistency_method}_summary_*.png: Summary of validation results")
    
    # Show DOF calculation for reference
    dof = calculate_dof_for_method(consistency_method)
    print(f"\nDOF Calculation for {consistency_method.upper()}:")
    if consistency_method == "nis3":
        print(f"  Proposition 3: DOF = Nz = T×nz + (T-1)×nx = 50×2 + 49×4 = {dof}")
    else:
        print(f"  Proposition 4: DOF = Nz - Nx = (T×nz + (T-1)×nx) - T×nx = {dof}")

if __name__ == "__main__":
    main() 
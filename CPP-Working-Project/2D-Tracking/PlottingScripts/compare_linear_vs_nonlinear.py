#!/usr/bin/env python3
"""
Compare linear vs nonlinear factor graph performance
Helps diagnose CNIS performance issues
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def load_bo_results(linear_path, nonlinear_path):
    """
    Load BO results from both linear and nonlinear systems
    """
    results = {}
    
    # Load linear results
    if os.path.exists(linear_path):
        with h5py.File(linear_path, 'r') as f:
            if 'trials' in f:
                data = f['trials'][:]
                results['linear'] = {
                    'Q': data[:, 0],
                    'R': data[:, 1],
                    'C': data[:, 2]
                }
            else:
                # Try individual arrays
                results['linear'] = {
                    'Q': f['q_values'][:],
                    'R': f['r_values'][:],
                    'C': f['objective_values'][:]
                }
    else:
        print(f"Warning: Linear BO results not found at {linear_path}")
        results['linear'] = None
    
    # Load nonlinear results
    if os.path.exists(nonlinear_path):
        with h5py.File(nonlinear_path, 'r') as f:
            results['nonlinear'] = {
                'Q': f['q_values'][:],
                'R': f['r_values'][:],
                'C': f['objective_values'][:]
            }
    else:
        print(f"Warning: Nonlinear BO results not found at {nonlinear_path}")
        results['nonlinear'] = None
    
    return results

def plot_comparison(results, save_plots=True):
    """
    Create comparison plots between linear and nonlinear systems
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear vs Nonlinear Factor Graph Performance Comparison', fontsize=16)
    
    colors = {'linear': 'blue', 'nonlinear': 'red'}
    markers = {'linear': 'o', 'nonlinear': 's'}
    
    for system, data in results.items():
        if data is None:
            continue
            
        color = colors[system]
        marker = markers[system]
        
        # Filter valid data
        valid = np.isfinite(data['C']) & (data['C'] < 1e5)
        Q = data['Q'][valid]
        R = data['R'][valid]
        C = data['C'][valid]
        
        if len(C) == 0:
            continue
        
        # Plot 1: Q vs R scatter
        axes[0, 0].scatter(Q, R, c=C, cmap='viridis', s=50, alpha=0.7, 
                          marker=marker, label=f'{system.capitalize()} trials')
        
        # Plot 2: C distribution histogram
        axes[0, 1].hist(C, bins=30, alpha=0.7, color=color, label=f'{system.capitalize()}', density=True)
        
        # Plot 3: C vs iteration
        iterations = np.arange(len(C))
        axes[1, 0].plot(iterations, C, color=color, alpha=0.7, label=f'{system.capitalize()}')
        
        # Plot 4: Running minimum
        running_min = np.minimum.accumulate(C)
        axes[1, 1].plot(iterations, running_min, color=color, linewidth=2, label=f'{system.capitalize()}')
    
    # Customize plots
    axes[0, 0].set_xlabel('Q (Process Noise)')
    axes[0, 0].set_ylabel('R (Measurement Noise)')
    axes[0, 0].set_title('Parameter Space Exploration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Consistency Metric (C)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Consistency Metrics')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Consistency Metric (C)')
    axes[1, 0].set_title('Consistency Metric vs Iteration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Running Minimum C')
    axes[1, 1].set_title('Convergence Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('../2D-Tracking/plots/linear_vs_nonlinear_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved as '../2D-Tracking/plots/linear_vs_nonlinear_comparison.png'")
    
    plt.show()

def print_comparison_statistics(results):
    """
    Print detailed comparison statistics
    """
    print("\n=== Linear vs Nonlinear Performance Comparison ===")
    
    for system, data in results.items():
        if data is None:
            print(f"\n{system.capitalize()} system: No data available")
            continue
            
        # Filter valid data
        valid = np.isfinite(data['C']) & (data['C'] < 1e5)
        Q = data['Q'][valid]
        R = data['R'][valid]
        C = data['C'][valid]
        
        if len(C) == 0:
            print(f"\n{system.capitalize()} system: No valid data")
            continue
        
        # Calculate statistics
        best_idx = np.argmin(C)
        best_Q = Q[best_idx]
        best_R = R[best_idx]
        best_C = C[best_idx]
        
        # Convergence analysis
        initial_C = C[0]
        final_C = C[-1]
        improvement = initial_C - final_C
        improvement_pct = (improvement / initial_C) * 100
        
        # Parameter ranges
        Q_range = np.max(Q) - np.min(Q)
        R_range = np.max(R) - np.min(R)
        C_range = np.max(C) - np.min(C)
        
        print(f"\n{system.capitalize()} System:")
        print(f"  Total trials: {len(C)}")
        print(f"  Best parameters: Q={best_Q:.4f}, R={best_R:.4f}, C={best_C:.4f}")
        print(f"  Parameter ranges: Q={Q_range:.4f}, R={R_range:.4f}, C={C_range:.4f}")
        print(f"  Initial C: {initial_C:.4f}")
        print(f"  Final C: {final_C:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        # CNIS-specific analysis
        if best_C > 5.0:  # High CNIS values might indicate issues
            print(f"  ⚠️  WARNING: High consistency metric ({best_C:.4f}) - possible CNIS issues")
        else:
            print(f"  ✅ Good consistency metric ({best_C:.4f})")

def analyze_cnis_issues(results):
    """
    Analyze potential CNIS performance issues
    """
    print("\n=== CNIS Performance Analysis ===")
    
    for system, data in results.items():
        if data is None:
            continue
            
        valid = np.isfinite(data['C']) & (data['C'] < 1e5)
        C = data['C'][valid]
        
        if len(C) == 0:
            continue
        
        # Analyze CNIS performance
        mean_C = np.mean(C)
        std_C = np.std(C)
        min_C = np.min(C)
        max_C = np.max(C)
        
        print(f"\n{system.capitalize()} System CNIS Analysis:")
        print(f"  Mean C: {mean_C:.4f} ± {std_C:.4f}")
        print(f"  C range: [{min_C:.4f}, {max_C:.4f}]")
        
        # Identify potential issues
        if mean_C > 10.0:
            print(f"  🔴 HIGH CNIS: Mean consistency metric is very high - possible model mismatch")
        elif mean_C > 6.0:
            print(f"  🟡 MODERATE CNIS: Elevated consistency metric - check noise parameters")
        else:
            print(f"  🟢 GOOD CNIS: Acceptable consistency metric")
        
        if std_C > 5.0:
            print(f"  ⚠️  HIGH VARIANCE: Large spread in CNIS values - inconsistent performance")
        
        # Check for convergence issues
        if max_C - min_C > 10.0:
            print(f"  ⚠️  POOR CONVERGENCE: Large range in CNIS values - optimization may not have converged")

def main():
    """
    Main function to compare linear vs nonlinear performance
    """
    # File paths
    linear_path = "../2D-Tracking/Saved_Data/2D_bayesopt_trials.h5"
    nonlinear_path = "../2D-Tracking/Saved_Data/2D_bayesopt_nonlinear_trials.h5"
    
    # Load results
    print("Loading BO results...")
    results = load_bo_results(linear_path, nonlinear_path)
    
    if results['linear'] is None and results['nonlinear'] is None:
        print("Error: No BO results found. Please run both linear and nonlinear BO first.")
        return
    
    # Print comparison statistics
    print_comparison_statistics(results)
    
    # Analyze CNIS issues
    analyze_cnis_issues(results)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(results, save_plots=True)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Plot Bayesian Optimization trials for nonlinear factor graph system
Visualizes Q, R, and consistency metric (C) from nonlinear BO results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
import yaml

def load_nonlinear_bo_trials(h5_path):
    """
    Load nonlinear BO trials from HDF5 file
    """
    if not os.path.exists(h5_path):
        print(f"Error: BO trials file not found: {h5_path}")
        print("Please run BO_Tracking_Test_Nonlinear first to generate nonlinear BO results.")
        return None, None, None
    
    with h5py.File(h5_path, 'r') as f:
        # Load individual arrays
        q_values = f['q_values'][:]
        r_values = f['r_values'][:]
        objective_values = f['objective_values'][:]
    
    return q_values, r_values, objective_values

def plot_3d_trials(Q, R, C, save_plots=True):
    """
    Create 3D scatter plot of BO trials
    """
    # Filter out invalid or penalized rows
    valid = np.isfinite(C) & (C < 1e5)
    Q_valid = Q[valid]
    R_valid = R[valid]
    C_valid = C[valid]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter all points
    sc = ax.scatter(Q_valid, R_valid, C_valid, c=C_valid, cmap='viridis', s=60, edgecolor='k')
    
    # Find the minimum C value and threshold
    min_C = np.min(C_valid)
    threshold = min_C + 0.1
    
    # Highlight points within threshold of the minimum C value
    close_mask = C_valid <= threshold
    ax.scatter(Q_valid[close_mask], R_valid[close_mask], C_valid[close_mask], 
              color='red', s=80, edgecolor='k', label=f'Within {threshold-min_C:.2f} of min(C)')
    
    ax.set_xlabel('Q (Process Noise Intensity)')
    ax.set_ylabel('R (Measurement Noise Variance)')
    ax.set_zlabel('C (Consistency Metric)')
    ax.set_title('Nonlinear BayesOpt Trials: Q, R, C (3D)')
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label('C (Consistency Metric)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('../2D-Tracking/plots/nonlinear_bayesopt_trials_3d.png', dpi=300, bbox_inches='tight')
        print("3D plot saved as '../2D-Tracking/plots/nonlinear_bayesopt_trials_3d.png'")
    
    plt.show()

def plot_2d_analysis(Q, R, C, save_plots=True):
    """
    Create 2D analysis plots of BO trials
    """
    # Filter out invalid or penalized rows
    valid = np.isfinite(C) & (C < 1e5)
    Q_valid = Q[valid]
    R_valid = R[valid]
    C_valid = C[valid]
    
    # Find the minimum C value and threshold
    min_C = np.min(C_valid)
    threshold = min_C + 0.1
    close_mask = C_valid <= threshold
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nonlinear BayesOpt Trials Analysis', fontsize=16)
    
    # Plot 1: Q vs R with best points highlighted
    axes[0, 0].scatter(Q_valid, R_valid, c=C_valid, cmap='viridis', s=50, alpha=0.7)
    axes[0, 0].scatter(Q_valid[close_mask], R_valid[close_mask], 
                      color='red', s=80, edgecolor='k', label=f'Best points (C ≤ {threshold:.2f})')
    axes[0, 0].set_xlabel('Q (Process Noise Intensity)')
    axes[0, 0].set_ylabel('R (Measurement Noise Variance)')
    axes[0, 0].set_title('Q vs R with Consistency Metric')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: C vs iteration
    iterations = np.arange(len(C_valid))
    axes[0, 1].plot(iterations, C_valid, 'b-', alpha=0.7, label='All trials')
    axes[0, 1].scatter(iterations[close_mask], C_valid[close_mask], 
                      color='red', s=50, label=f'Best trials (C ≤ {threshold:.2f})')
    axes[0, 1].axhline(y=min_C, color='g', linestyle='--', label=f'Min C = {min_C:.3f}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Consistency Metric (C)')
    axes[0, 1].set_title('Consistency Metric vs Iteration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q vs C
    axes[1, 0].scatter(Q_valid, C_valid, c=R_valid, cmap='plasma', s=50, alpha=0.7)
    axes[1, 0].scatter(Q_valid[close_mask], C_valid[close_mask], 
                      color='red', s=80, edgecolor='k', label=f'Best points')
    axes[1, 0].set_xlabel('Q (Process Noise Intensity)')
    axes[1, 0].set_ylabel('Consistency Metric (C)')
    axes[1, 0].set_title('Q vs C (colored by R)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: R vs C
    axes[1, 1].scatter(R_valid, C_valid, c=Q_valid, cmap='plasma', s=50, alpha=0.7)
    axes[1, 1].scatter(R_valid[close_mask], C_valid[close_mask], 
                      color='red', s=80, edgecolor='k', label=f'Best points')
    axes[1, 1].set_xlabel('R (Measurement Noise Variance)')
    axes[1, 1].set_ylabel('Consistency Metric (C)')
    axes[1, 1].set_title('R vs C (colored by Q)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('../2D-Tracking/plots/nonlinear_bayesopt_trials_2d.png', dpi=300, bbox_inches='tight')
        print("2D analysis plots saved as '../2D-Tracking/plots/nonlinear_bayesopt_trials_2d.png'")
    
    plt.show()

def plot_convergence_analysis(Q, R, C, save_plots=True):
    """
    Analyze convergence of the BO optimization
    """
    # Filter out invalid or penalized rows
    valid = np.isfinite(C) & (C < 1e5)
    Q_valid = Q[valid]
    R_valid = R[valid]
    C_valid = C[valid]
    
    # Calculate running minimum
    running_min = np.minimum.accumulate(C_valid)
    
    # Find best parameters
    best_idx = np.argmin(C_valid)
    best_Q = Q_valid[best_idx]
    best_R = R_valid[best_idx]
    best_C = C_valid[best_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nonlinear BayesOpt Convergence Analysis', fontsize=16)
    
    # Plot 1: Convergence of objective function
    iterations = np.arange(len(C_valid))
    axes[0, 0].plot(iterations, C_valid, 'b-', alpha=0.5, label='All trials')
    axes[0, 0].plot(iterations, running_min, 'r-', linewidth=2, label='Running minimum')
    axes[0, 0].scatter(best_idx, best_C, color='red', s=100, zorder=5, 
                      label=f'Best: C={best_C:.3f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Consistency Metric (C)')
    axes[0, 0].set_title('Objective Function Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence
    axes[0, 1].plot(iterations, Q_valid, 'b-', alpha=0.7, label='Q trials')
    axes[0, 1].plot(iterations, R_valid, 'g-', alpha=0.7, label='R trials')
    axes[0, 1].axhline(y=best_Q, color='b', linestyle='--', alpha=0.7, label=f'Best Q = {best_Q:.3f}')
    axes[0, 1].axhline(y=best_R, color='g', linestyle='--', alpha=0.7, label=f'Best R = {best_R:.3f}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].set_title('Parameter Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Parameter space exploration
    axes[1, 0].scatter(Q_valid, R_valid, c=iterations, cmap='viridis', s=50, alpha=0.7)
    axes[1, 0].scatter(best_Q, best_R, color='red', s=200, marker='*', zorder=5, 
                      label=f'Best: Q={best_Q:.3f}, R={best_R:.3f}')
    axes[1, 0].set_xlabel('Q (Process Noise Intensity)')
    axes[1, 0].set_ylabel('R (Measurement Noise Variance)')
    axes[1, 0].set_title('Parameter Space Exploration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Improvement over iterations
    improvement = running_min[0] - running_min
    axes[1, 1].plot(iterations, improvement, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Improvement in C')
    axes[1, 1].set_title('Cumulative Improvement')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('../2D-Tracking/plots/nonlinear_bayesopt_convergence.png', dpi=300, bbox_inches='tight')
        print("Convergence analysis saved as '../2D-Tracking/plots/nonlinear_bayesopt_convergence.png'")
    
    plt.show()

def print_bo_statistics(Q, R, C):
    """
    Print statistics about the BO trials
    """
    # Filter out invalid or penalized rows
    valid = np.isfinite(C) & (C < 1e5)
    Q_valid = Q[valid]
    R_valid = R[valid]
    C_valid = C[valid]
    
    print("\n=== Nonlinear BayesOpt Statistics ===")
    print(f"Total trials: {len(C_valid)}")
    print(f"Valid trials: {np.sum(valid)}")
    
    # Find best parameters
    best_idx = np.argmin(C_valid)
    best_Q = Q_valid[best_idx]
    best_R = R_valid[best_idx]
    best_C = C_valid[best_idx]
    
    print(f"\nBest parameters found:")
    print(f"  Q (Process Noise Intensity): {best_Q:.4f}")
    print(f"  R (Measurement Noise Variance): {best_R:.4f}")
    print(f"  C (Consistency Metric): {best_C:.4f}")
    
    print(f"\nParameter ranges explored:")
    print(f"  Q range: [{np.min(Q_valid):.4f}, {np.max(Q_valid):.4f}]")
    print(f"  R range: [{np.min(R_valid):.4f}, {np.max(R_valid):.4f}]")
    print(f"  C range: [{np.min(C_valid):.4f}, {np.max(C_valid):.4f}]")
    
    # Calculate improvement
    initial_C = C_valid[0]
    final_C = C_valid[-1]
    improvement = initial_C - final_C
    improvement_pct = (improvement / initial_C) * 100
    
    print(f"\nOptimization improvement:")
    print(f"  Initial C: {initial_C:.4f}")
    print(f"  Final C: {final_C:.4f}")
    print(f"  Absolute improvement: {improvement:.4f}")
    print(f"  Percentage improvement: {improvement_pct:.2f}%")

def plot_filtered_qr(Q, R, C, yaml_path='../scenario_nonlinear.yaml'):
    try:
        with open(yaml_path, 'r') as yf:
            cfg = yaml.safe_load(yf)
        q_true = float(cfg['Data_Generation']['q'])
        R_true = float(cfg['Data_Generation']['meas_noise_var'])
        abs_tol = float(os.environ.get('BO_FILTER_ABS_TOL', '0.01'))  # absolute window, default 0.01
        q_min, q_max = q_true - abs_tol, q_true + abs_tol
        R_min, R_max = R_true - abs_tol, R_true + abs_tol

        valid = np.isfinite(C) & (C < 1e5)
        Qv = Q[valid]; Rv = R[valid]; Cv = C[valid]
        mask = (Qv >= q_min) & (Qv <= q_max) & (Rv >= R_min) & (Rv <= R_max)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(Qv[mask], Rv[mask], c=Cv[mask], cmap='viridis', s=60, edgecolor='k')
        plt.colorbar(sc, label='C (Consistency Metric)')
        plt.scatter([q_true], [R_true], color='red', s=120, marker='*', label='True (q, R)')
        plt.xlabel('Q (Process Noise Intensity)')
        plt.ylabel('R (Measurement Noise Variance)')
        plt.title(f'Filtered Q–R near true values (±{abs_tol})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipped filtered 2D plot: {e}")

def main():
    """
    Main function to load and plot nonlinear BO trials
    """
    # File path for nonlinear BO trials
    h5_path = "../2D-Tracking/Saved_Data/2D_bayesopt_nonlinear_trials.h5"
    
    # Load trials
    print("Loading nonlinear BO trials...")
    Q, R, C = load_nonlinear_bo_trials(h5_path)
    
    if Q is None:
        return
    
    # Print statistics
    print_bo_statistics(Q, R, C)
    
    # Create plots
    print("\nCreating 3D trials plot...")
    plot_3d_trials(Q, R, C, save_plots=True)
    
    print("\nCreating 2D analysis plots...")
    plot_2d_analysis(Q, R, C, save_plots=True)

    print("\nCreating filtered 2D Q–R plot near ground truth...")
    plot_filtered_qr(Q, R, C)
    
    print("\nCreating convergence analysis...")
    plot_convergence_analysis(Q, R, C, save_plots=True)
    
    print("\nAll plots completed!")

if __name__ == "__main__":
    main() 
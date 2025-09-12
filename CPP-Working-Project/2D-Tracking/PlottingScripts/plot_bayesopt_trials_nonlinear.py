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
from matplotlib.colors import LogNorm

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

def plot_topdown_heatmap(Q, R, C, save_plots=True, bins=60):
    """
    Top-down 2D scatter of C over (Q, R) using a logarithmic color scale for C.
    """
    valid = np.isfinite(C) & (C < 1e5)
    Qv = Q[valid]
    Rv = R[valid]
    Cv = C[valid]

    if Qv.size == 0:
        print("No valid data to plot heatmap.")
        return

    # Ensure strictly positive values for LogNorm
    positive_mask = Cv > 0
    if not np.any(positive_mask):
        # Fallback to linear scale if no positive values
        norm = None
        vmin, vmax = np.min(Cv), np.max(Cv)
    else:
        vmin = np.max([np.min(Cv[positive_mask]), 1e-9])
        vmax = np.max(Cv)
        norm = LogNorm(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Qv, Rv, c=Cv, cmap='viridis', norm=norm, s=60, edgecolor='k')
    cbar = plt.colorbar(sc)
    cbar.set_label('C (Consistency Metric)')

    # Mark global best point
    best_idx = np.argmin(Cv)
    best_q = Qv[best_idx]
    best_r = Rv[best_idx]
    best_c = Cv[best_idx]
    best_sigma = np.sqrt(best_r) if best_r >= 0 else float('nan')
    plt.scatter([best_q], [best_r], color='red', s=120, marker='*', edgecolor='k', label='Best')

    plt.xlabel('Q (Process Noise Intensity)')
    plt.ylabel('R (Measurement Noise Variance)')
    plt.title(f'CNIS over Q(V) and R(σ²): min CNIS={best_c:.4g} at Q={best_q:.4g}, R={best_r:.4g} (σ={best_sigma:.4g})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_plots:
        plt.savefig('../2D-Tracking/plots/nonlinear_bayesopt_trials_heatmap.png', dpi=300, bbox_inches='tight')
        print("Heatmap saved as '../2D-Tracking/plots/nonlinear_bayesopt_trials_heatmap.png'")

    plt.show()

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
    h5_path = "../2D-Tracking/Saved_Data/2D_bayesopt_nonlinear_trials_run1.h5"
    
    # Load trials
    print("Loading nonlinear BO trials...")
    Q, R, C = load_nonlinear_bo_trials(h5_path)
    
    if Q is None:
        return
    
    # Create plots (3D + top-down heatmap)
    print("\nCreating 3D trials plot...")
    plot_3d_trials(Q, R, C, save_plots=True)

    print("\nCreating top-down 2D heatmap...")
    plot_topdown_heatmap(Q, R, C, save_plots=True)
    
    print("\nDone.")

if __name__ == "__main__":
    main() 
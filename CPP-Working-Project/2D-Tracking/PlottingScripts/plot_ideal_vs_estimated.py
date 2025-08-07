#!/usr/bin/env python3
"""
Ideal vs Estimated Trajectory Comparison
========================================

This script creates a focused plot comparing the ideal trajectory 
with the estimated trajectory using optimal parameters from Bayesian Optimization.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def load_demo_results(filename):
    """Load demonstration results from HDF5 file."""
    print(f"Loading results from: {filename}")
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return None
    
    try:
        with h5py.File(filename, 'r') as f:
            # Load datasets
            true_states = np.array(f['true_states'])
            estimated_states = np.array(f['estimated_states'])
            measurements = np.array(f['measurements'])
            chi2_values = np.array(f['chi2_values'])
            
            # Load metadata
            tuned_q = f.attrs['tuned_q']
            tuned_R = f.attrs['tuned_R']
            dt = f.attrs['dt']
            best_cnees = f.attrs['best_cnees']
            
            print(f"Loaded {true_states.shape[0]} trajectories, each with {true_states.shape[1]} time steps")
            print(f"Optimal parameters: q={tuned_q:.4f}, R={tuned_R:.4f}, CNEES={best_cnees:.6f}")
            
            return {
                'true_states': true_states,
                'estimated_states': estimated_states,
                'measurements': measurements,
                'chi2_values': chi2_values,
                'tuned_q': tuned_q,
                'tuned_R': tuned_R,
                'dt': dt,
                'best_cnees': best_cnees
            }
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def plot_ideal_vs_estimated(data, run_idx=0, save_path="../2D-Tracking/plots"):
    """Create a focused plot comparing ideal vs estimated trajectory."""
    
    # Extract data for the specified run
    ideal_pos = data['true_states'][run_idx, :, :2]  # [x, y] positions (ideal trajectory)
    estimated_pos = data['estimated_states'][run_idx, :, :2]  # [x, y] positions (estimated)
    measurements = data['measurements'][run_idx, :, :2]  # [x, y] measurements
    chi2 = data['chi2_values'][run_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: 2D Trajectory Comparison
    ax1.plot(ideal_pos[:, 0], ideal_pos[:, 1], 'g-', linewidth=3, 
             label='Ideal Trajectory', alpha=0.9)
    ax1.plot(estimated_pos[:, 0], estimated_pos[:, 1], 'b--', linewidth=2.5, 
             label='Estimated Trajectory (Optimal Parameters)', alpha=0.8)
    ax1.scatter(measurements[:, 0], measurements[:, 1], c='red', s=15, alpha=0.5, 
                label='Noisy Measurements', zorder=5)
    
    # Mark start and end points
    ax1.scatter(ideal_pos[0, 0], ideal_pos[0, 1], c='green', s=120, marker='s', 
                label='Start (Ideal)', zorder=10, edgecolors='black', linewidth=1)
    ax1.scatter(ideal_pos[-1, 0], ideal_pos[-1, 1], c='green', s=120, marker='^', 
                label='End (Ideal)', zorder=10, edgecolors='black', linewidth=1)
    ax1.scatter(estimated_pos[0, 0], estimated_pos[0, 1], c='blue', s=100, marker='o', 
                label='Start (Estimated)', zorder=10, edgecolors='black', linewidth=1)
    ax1.scatter(estimated_pos[-1, 0], estimated_pos[-1, 1], c='blue', s=100, marker='D', 
                label='End (Estimated)', zorder=10, edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'Ideal vs Estimated Trajectory (Run {run_idx+1})\n' + 
                  f'Optimal Parameters: q={data["tuned_q"]:.4f}, R={data["tuned_R"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Position Error Over Time
    time_steps = np.arange(len(ideal_pos)) * data['dt']
    pos_errors = np.linalg.norm(ideal_pos - estimated_pos, axis=1)
    
    ax2.plot(time_steps, pos_errors, 'r-', linewidth=2.5, label='Position Error')
    ax2.fill_between(time_steps, 0, pos_errors, alpha=0.3, color='red')
    
    mean_error = np.mean(pos_errors)
    max_error = np.max(pos_errors)
    final_error = pos_errors[-1]
    
    ax2.axhline(y=mean_error, color='orange', linestyle='--', linewidth=2, 
                label=f'Mean Error = {mean_error:.3f}')
    ax2.axhline(y=max_error, color='purple', linestyle=':', linewidth=2, 
                label=f'Max Error = {max_error:.3f}')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position Error', fontsize=12)
    ax2.set_title(f'Position Error Over Time\n' + 
                  f'Chi² = {chi2:.2f}, Final Error = {final_error:.3f}', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with performance metrics
    performance_text = f'Performance Metrics:\n' + \
                      f'• Mean Position Error: {mean_error:.3f}\n' + \
                      f'• Max Position Error: {max_error:.3f}\n' + \
                      f'• Final Position Error: {final_error:.3f}\n' + \
                      f'• Chi-squared Value: {chi2:.2f}\n' + \
                      f'• Best CNEES: {data["best_cnees"]:.6f}'
    
    ax2.text(0.02, 0.98, performance_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/ideal_vs_estimated_trajectory_run_{run_idx+1}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot: {filename}")
    return fig

def plot_multiple_runs_comparison(data, num_runs=3, save_path="../2D-Tracking/plots"):
    """Create a comparison plot showing multiple runs."""
    
    fig, axes = plt.subplots(1, num_runs, figsize=(6*num_runs, 5))
    if num_runs == 1:
        axes = [axes]
    
    for run_idx in range(num_runs):
        if run_idx >= data['true_states'].shape[0]:
            break
            
        ideal_pos = data['true_states'][run_idx, :, :2]
        estimated_pos = data['estimated_states'][run_idx, :, :2]
        measurements = data['measurements'][run_idx, :, :2]
        chi2 = data['chi2_values'][run_idx]
        
        ax = axes[run_idx]
        
        ax.plot(ideal_pos[:, 0], ideal_pos[:, 1], 'g-', linewidth=2.5, 
                label='Ideal', alpha=0.9)
        ax.plot(estimated_pos[:, 0], estimated_pos[:, 1], 'b--', linewidth=2, 
                label='Estimated', alpha=0.8)
        ax.scatter(measurements[:, 0], measurements[:, 1], c='red', s=10, alpha=0.4, 
                   label='Measurements', zorder=5)
        
        # Mark start and end points
        ax.scatter(ideal_pos[0, 0], ideal_pos[0, 1], c='green', s=80, marker='s', 
                   zorder=10, edgecolors='black', linewidth=1)
        ax.scatter(ideal_pos[-1, 0], ideal_pos[-1, 1], c='green', s=80, marker='^', 
                   zorder=10, edgecolors='black', linewidth=1)
        ax.scatter(estimated_pos[0, 0], estimated_pos[0, 1], c='blue', s=60, marker='o', 
                   zorder=10, edgecolors='black', linewidth=1)
        ax.scatter(estimated_pos[-1, 0], estimated_pos[-1, 1], c='blue', s=60, marker='D', 
                   zorder=10, edgecolors='black', linewidth=1)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Run {run_idx+1}\nChi² = {chi2:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle(f'Ideal vs Estimated Trajectories - Multiple Runs\n' + 
                 f'Optimal Parameters: q={data["tuned_q"]:.4f}, R={data["tuned_R"]:.4f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/ideal_vs_estimated_multiple_runs.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved plot: {filename}")
    return fig

def main():
    """Main function to generate the ideal vs estimated trajectory plots."""
    print("=== Ideal vs Estimated Trajectory Comparison ===")
    
    # Load data
    data_file = "../2D-Tracking/Saved_Data/2D_tuned_demo_results.h5"
    data = load_demo_results(data_file)
    
    if data is None:
        return
    
    print(f"\nGenerating ideal vs estimated trajectory plots...")
    
    # Create detailed plot for the first run
    print("Creating detailed comparison for run 1...")
    plot_ideal_vs_estimated(data, run_idx=0)
    
    # Create comparison plot for multiple runs
    num_runs = min(3, data['true_states'].shape[0])
    print(f"Creating comparison plot for {num_runs} runs...")
    plot_multiple_runs_comparison(data, num_runs=num_runs)
    
    print("\n=== Plotting Complete ===")
    print("Generated plots:")
    print("  - ideal_vs_estimated_trajectory_run_1.png: Detailed comparison")
    print("  - ideal_vs_estimated_multiple_runs.png: Multiple runs comparison")
    print("\nSummary:")
    print(f"  Optimal process noise intensity (q): {data['tuned_q']:.4f}")
    print(f"  Optimal measurement noise variance (R): {data['tuned_R']:.4f}")
    print(f"  Best CNEES objective: {data['best_cnees']:.6f}")

if __name__ == "__main__":
    main() 
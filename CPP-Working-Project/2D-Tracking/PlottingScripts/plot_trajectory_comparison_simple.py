#!/usr/bin/env python3
"""
Simple Trajectory Comparison Visualization
==========================================

This script creates trajectory comparison plots using only matplotlib and pandas.
No external dependencies like seaborn required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Set matplotlib style
plt.style.use('classic')

def load_trajectory_data(plots_dir):
    """Load trajectory data from CSV file, preferring fixed version."""
    
    # Try fixed version first
    fixed_file = os.path.join(plots_dir, "trajectory_comparison_data_fixed.csv")
    original_file = os.path.join(plots_dir, "trajectory_comparison_data.csv")
    
    if os.path.exists(fixed_file):
        try:
            data = pd.read_csv(fixed_file)
            print(f"✓ Loaded FIXED trajectory data: {len(data)} time steps")
            print("  (Model-consistent: no accelerations in data generation)")
            return data, "fixed"
        except Exception as e:
            print(f"Error loading fixed data: {e}")
    
    if os.path.exists(original_file):
        try:
            data = pd.read_csv(original_file)
            print(f"⚠ Loaded ORIGINAL trajectory data: {len(data)} time steps")
            print("  (Model mismatch: data has accelerations, factor graph assumes constant velocity)")
            return data, "original"
        except Exception as e:
            print(f"Error loading original data: {e}")
    
    print("Error: Could not find trajectory data files!")
    print("Please run one of these C++ data generation programs first:")
    print("  cd build && ./generate_trajectory_comparison_data_fixed  (recommended)")
    print("  cd build && ./generate_trajectory_comparison_data        (original with model mismatch)")
    return None, None

def calculate_errors(data):
    """Calculate position and velocity errors for all estimators."""
    errors = {}
    
    # Position errors
    errors['tuned_pos'] = np.sqrt((data['true_x'] - data['tuned_x'])**2 + 
                                 (data['true_y'] - data['tuned_y'])**2)
    errors['bad1_pos'] = np.sqrt((data['true_x'] - data['bad1_x'])**2 + 
                                (data['true_y'] - data['bad1_y'])**2)
    errors['bad2_pos'] = np.sqrt((data['true_x'] - data['bad2_x'])**2 + 
                                (data['true_y'] - data['bad2_y'])**2)
    
    # Velocity errors
    errors['tuned_vel'] = np.sqrt((data['true_vx'] - data['tuned_vx'])**2 + 
                                 (data['true_vy'] - data['tuned_vy'])**2)
    errors['bad1_vel'] = np.sqrt((data['true_vx'] - data['bad1_vx'])**2 + 
                                (data['true_vy'] - data['bad1_vy'])**2)
    errors['bad2_vel'] = np.sqrt((data['true_vx'] - data['bad2_vx'])**2 + 
                                (data['true_vy'] - data['bad2_vy'])**2)
    
    # Measurement errors
    errors['meas_pos'] = np.sqrt((data['true_x'] - data['meas_x'])**2 + 
                                (data['true_y'] - data['meas_y'])**2)
    
    return errors

def create_trajectory_plots(data, errors, save_path, data_type="unknown"):
    """Create comprehensive trajectory comparison plots."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    title_suffix = " (Model-Consistent)" if data_type == "fixed" else " (Model Mismatch!)" if data_type == "original" else ""
    fig.suptitle(f'Factor Graph Performance: Tuned vs. Mistuned Parameters{title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # Colors for consistency
    colors = {
        'true': 'green',
        'tuned': 'blue', 
        'bad1': 'red',
        'bad2': 'magenta',
        'meas': 'orange'
    }
    
    # Plot 1: Main 2D trajectory comparison
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data['true_x'], data['true_y'], '-', color=colors['true'], linewidth=3, 
             label='True Trajectory', alpha=0.9)
    ax1.plot(data['tuned_x'], data['tuned_y'], '--', color=colors['tuned'], linewidth=2, 
             label='Tuned Parameters', alpha=0.8)
    ax1.plot(data['bad1_x'], data['bad1_y'], ':', color=colors['bad1'], linewidth=2, 
             label='Bad Params (Low Q, High R)', alpha=0.7)
    ax1.plot(data['bad2_x'], data['bad2_y'], '-.', color=colors['bad2'], linewidth=2, 
             label='Bad Params (High Q, Low R)', alpha=0.7)
    
    # Add measurement points (subsample for clarity)
    step = max(1, len(data) // 15)
    ax1.scatter(data['meas_x'][::step], data['meas_y'][::step], 
               c=colors['meas'], s=25, alpha=0.6, label='Measurements', zorder=3)
    
    # Mark start and end
    ax1.plot(data['true_x'].iloc[0], data['true_y'].iloc[0], 'go', markersize=8, label='Start')
    ax1.plot(data['true_x'].iloc[-1], data['true_y'].iloc[-1], 'rs', markersize=8, label='End')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Position error over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(data['time'], errors['tuned_pos'], '-', color=colors['tuned'], linewidth=2, 
             label='Tuned')
    ax2.plot(data['time'], errors['bad1_pos'], '--', color=colors['bad1'], linewidth=2, 
             label='Bad Set 1')
    ax2.plot(data['time'], errors['bad2_pos'], ':', color=colors['bad2'], linewidth=2, 
             label='Bad Set 2')
    ax2.plot(data['time'], errors['meas_pos'], '-', color=colors['meas'], alpha=0.5, 
             linewidth=1, label='Measurement Error')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position Error')
    ax2.set_title('Position Error vs. Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: X-coordinate tracking
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(data['time'], data['true_x'], '-', color=colors['true'], linewidth=3, 
             label='True X')
    ax3.plot(data['time'], data['tuned_x'], '--', color=colors['tuned'], linewidth=2, 
             label='Tuned X')
    ax3.plot(data['time'], data['bad1_x'], ':', color=colors['bad1'], linewidth=2, 
             label='Bad Set 1 X')
    ax3.plot(data['time'], data['bad2_x'], '-.', color=colors['bad2'], linewidth=2, 
             label='Bad Set 2 X')
    ax3.scatter(data['time'][::8], data['meas_x'][::8], c=colors['meas'], s=8, alpha=0.6)
    
    ax3.set_title('X-Coordinate Tracking')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('X Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y-coordinate tracking  
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(data['time'], data['true_y'], '-', color=colors['true'], linewidth=3, 
             label='True Y')
    ax4.plot(data['time'], data['tuned_y'], '--', color=colors['tuned'], linewidth=2, 
             label='Tuned Y')
    ax4.plot(data['time'], data['bad1_y'], ':', color=colors['bad1'], linewidth=2, 
             label='Bad Set 1 Y')
    ax4.plot(data['time'], data['bad2_y'], '-.', color=colors['bad2'], linewidth=2, 
             label='Bad Set 2 Y')
    ax4.scatter(data['time'][::8], data['meas_y'][::8], c=colors['meas'], s=8, alpha=0.6)
    
    ax4.set_title('Y-Coordinate Tracking')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Y Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity error over time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(data['time'], errors['tuned_vel'], '-', color=colors['tuned'], linewidth=2, 
             label='Tuned')
    ax5.plot(data['time'], errors['bad1_vel'], '--', color=colors['bad1'], linewidth=2, 
             label='Bad Set 1')
    ax5.plot(data['time'], errors['bad2_vel'], ':', color=colors['bad2'], linewidth=2, 
             label='Bad Set 2')
    
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Velocity Error')
    ax5.set_title('Velocity Error vs. Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error statistics summary
    ax6 = fig.add_subplot(gs[2, 0])
    categories = ['Tuned\nParams', 'Bad Set 1\n(Low Q)', 'Bad Set 2\n(High Q)']
    
    avg_pos_errors = [
        np.mean(errors['tuned_pos']),
        np.mean(errors['bad1_pos']),
        np.mean(errors['bad2_pos'])
    ]
    avg_vel_errors = [
        np.mean(errors['tuned_vel']),
        np.mean(errors['bad1_vel']),
        np.mean(errors['bad2_vel'])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, avg_pos_errors, width, label='Position Error', 
                    alpha=0.8, color='skyblue')
    bars2 = ax6.bar(x + width/2, avg_vel_errors, width, label='Velocity Error', 
                    alpha=0.8, color='lightcoral')
    
    ax6.set_xlabel('Parameter Set')
    ax6.set_ylabel('Average Error')
    ax6.set_title('Average Error Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 7: Cumulative squared error
    ax7 = fig.add_subplot(gs[2, 1])
    
    cum_tuned = np.cumsum(errors['tuned_pos']**2)
    cum_bad1 = np.cumsum(errors['bad1_pos']**2)
    cum_bad2 = np.cumsum(errors['bad2_pos']**2)
    
    ax7.plot(data['time'], cum_tuned, '-', color=colors['tuned'], linewidth=2, label='Tuned')
    ax7.plot(data['time'], cum_bad1, '--', color=colors['bad1'], linewidth=2, label='Bad Set 1')
    ax7.plot(data['time'], cum_bad2, ':', color=colors['bad2'], linewidth=2, label='Bad Set 2')
    
    ax7.set_title('Cumulative Squared Error')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Cumulative Error²')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance improvement factors
    ax8 = fig.add_subplot(gs[2, 2])
    
    tuned_avg = np.mean(errors['tuned_pos'])
    bad1_avg = np.mean(errors['bad1_pos'])
    bad2_avg = np.mean(errors['bad2_pos'])
    
    improvement_factors = [1.0, bad1_avg / tuned_avg, bad2_avg / tuned_avg]
    param_sets = ['Tuned\n(Baseline)', 'Bad Set 1\n(Error Factor)', 'Bad Set 2\n(Error Factor)']
    
    bars = ax8.bar(param_sets, improvement_factors, 
                   color=['green', 'red', 'magenta'], alpha=0.7)
    
    ax8.set_ylabel('Error Factor (vs. Tuned)')
    ax8.set_title('Performance Degradation')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, improvement_factors):
        height = bar.get_height()
        ax8.annotate(f'{val:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory comparison plot: {save_path}")
    return fig

def print_summary_statistics(data, errors):
    """Print comprehensive summary statistics."""
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate improvement factors
    tuned_avg_pos = np.mean(errors['tuned_pos'])
    bad1_avg_pos = np.mean(errors['bad1_pos'])
    bad2_avg_pos = np.mean(errors['bad2_pos'])
    
    improvement1 = bad1_avg_pos / tuned_avg_pos
    improvement2 = bad2_avg_pos / tuned_avg_pos
    
    print(f"\nPosition Error Performance:")
    print(f"  Tuned Parameters:      {tuned_avg_pos:.6f}")
    print(f"  Bad Set 1 (Low Q, High R): {bad1_avg_pos:.6f}")
    print(f"  Bad Set 2 (High Q, Low R): {bad2_avg_pos:.6f}")
    print(f"\nImprovement Factors:")
    print(f"  Tuned vs Bad Set 1: {improvement1:.1f}x better")
    print(f"  Tuned vs Bad Set 2: {improvement2:.1f}x better")
    
    print(f"\nFinal Position Errors:")
    print(f"  Tuned:     {errors['tuned_pos'].iloc[-1]:.6f}")
    print(f"  Bad Set 1: {errors['bad1_pos'].iloc[-1]:.6f}")
    print(f"  Bad Set 2: {errors['bad2_pos'].iloc[-1]:.6f}")
    
    print(f"\nTrajectory Statistics:")
    print(f"  Time steps: {len(data)}")
    print(f"  Total time: {data['time'].iloc[-1]:.2f}")
    
    # Calculate distance traveled
    dx = np.diff(data['true_x'])
    dy = np.diff(data['true_y'])
    distances = np.sqrt(dx**2 + dy**2)
    total_distance = np.sum(distances)
    print(f"  Distance traveled: {total_distance:.2f}")

def main():
    """Main plotting function."""
    
    # Create output directory
    plots_dir = "../2D-Tracking/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data (try fixed version first)
    result = load_trajectory_data(plots_dir)
    
    if result[0] is None:
        return
    
    data, data_type = result
    
    # Calculate errors
    errors = calculate_errors(data)
    
    # Create comprehensive plot with appropriate filename
    filename_suffix = "_fixed" if data_type == "fixed" else "_original"
    plot_path = os.path.join(plots_dir, f"trajectory_comparison_comprehensive{filename_suffix}.png")
    fig = create_trajectory_plots(data, errors, plot_path, data_type)
    
    # Print summary
    print_summary_statistics(data, errors)
    
    print(f"\n✓ Visualization complete!")
    print(f"Generated plot: {plot_path}")
    if data_type == "original":
        print("\n⚠ WARNING: This plot shows MODEL MISMATCH effects!")
        print("   Data generated with accelerations, but factor graph assumes constant velocity.")
        print("   For true parameter comparison, run the fixed version:")
        print("   cd build && ./generate_trajectory_comparison_data_fixed")
    
    # Also try to display (might not work in headless environment)
    try:
        plt.show()
    except:
        print("Note: Could not display plot (headless environment)")

if __name__ == "__main__":
    main() 
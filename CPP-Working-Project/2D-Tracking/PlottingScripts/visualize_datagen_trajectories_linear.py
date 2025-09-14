#!/usr/bin/env python3
"""
Visualize the first 2 trajectories from HDF5 files
Shows states and measurements for the first 2 Monte Carlo runs
Also supports plotting a single selected trajectory in a separate 8x6 figure
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def load_h5_trajectories(states_file, measurements_file, num_trajectories=2):
    """
    Load the first num_trajectories from HDF5 files
    """
    with h5py.File(states_file, 'r') as f:
        states = f['states'][:num_trajectories, :, :]  # Shape: (num_traj, N, 4)
    
    with h5py.File(measurements_file, 'r') as f:
        measurements = f['measurements'][:num_trajectories, :, :]  # Shape: (num_traj, N, 2)
    
    return states, measurements


def plot_trajectories(states, measurements, dt=0.1, save_plots=True):
    """
    Plot the first 2 trajectories with states and measurements
    """
    num_trajectories = states.shape[0]
    N = states.shape[1]
    time = np.arange(N) * dt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('First 2 Trajectories: States and Measurements', fontsize=16)
    
    colors = ['blue', 'red']
    labels = ['Trajectory 1', 'Trajectory 2']
    
    for traj_idx in range(num_trajectories):
        color = colors[traj_idx]
        label = labels[traj_idx]
        
        # Extract state components
        x_pos = states[traj_idx, :, 0]  # x position
        y_pos = states[traj_idx, :, 1]  # y position
        x_vel = states[traj_idx, :, 2]  # x velocity
        y_vel = states[traj_idx, :, 3]  # y velocity
        
        # Extract measurements
        x_meas = measurements[traj_idx, :, 0]  # x measurement
        y_meas = measurements[traj_idx, :, 1]  # y measurement
        
        # Plot 1: Position trajectory (x vs y)
        axes[0, 0].plot(x_pos, y_pos, color=color, linewidth=2, label=f'{label} - True')
        axes[0, 0].scatter(x_meas, y_meas, color=color, alpha=0.6, s=20, 
                          label=f'{label} - Measurements', marker='o')
        axes[0, 0].scatter(x_pos[0], y_pos[0], color=color, s=100, marker='s', 
                          label=f'{label} - Start', zorder=5)
        axes[0, 0].scatter(x_pos[-1], y_pos[-1], color=color, s=100, marker='^', 
                          label=f'{label} - End', zorder=5)
        
        # Plot 2: Position vs time
        axes[0, 1].plot(time, x_pos, color=color, linewidth=2, label=f'{label} - x pos')
        axes[0, 1].plot(time, y_pos, color=color, linewidth=2, linestyle='--', 
                       label=f'{label} - y pos')
        axes[0, 1].scatter(time, x_meas, color=color, alpha=0.6, s=15, marker='o')
        axes[0, 1].scatter(time, y_meas, color=color, alpha=0.6, s=15, marker='s')
        
        # Plot 3: Velocity vs time
        axes[1, 0].plot(time, x_vel, color=color, linewidth=2, label=f'{label} - x vel')
        axes[1, 0].plot(time, y_vel, color=color, linewidth=2, linestyle='--', 
                       label=f'{label} - y vel')
        
        # Plot 4: Speed vs time
        speed = np.sqrt(x_vel**2 + y_vel**2)
        axes[1, 1].plot(time, speed, color=color, linewidth=2, label=f'{label} - Speed')
    
    # Customize plots
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Position Trajectory (X vs Y)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].set_title('Position vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].set_title('Velocity vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed')
    axes[1, 1].set_title('Speed vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('../H5_Files/trajectory_visualization.png', dpi=300, bbox_inches='tight')
        print("Plot saved as '../H5_Files/trajectory_visualization.png'")
    
    plt.show()


def plot_single_trajectory(states, measurements, traj_idx=0, save_plot=True):
    """
    Plot a single trajectory (XY with measurements) in an 8x6 figure
    """
    x_pos = states[traj_idx, :, 0]
    y_pos = states[traj_idx, :, 1]
    x_meas = measurements[traj_idx, :, 0]
    y_meas = measurements[traj_idx, :, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x_pos, y_pos, color='blue', linewidth=2, label='True')
    plt.scatter(x_meas, y_meas, color='orange', s=15, alpha=0.7, label='Measurements')
    plt.scatter(x_pos[0], y_pos[0], color='green', s=80, marker='s', label='Start', zorder=5)
    plt.scatter(x_pos[-1], y_pos[-1], color='red', s=80, marker='^', label='End', zorder=5)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Linear Trajectory (run {traj_idx})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plot:
        os.makedirs('../2D-Tracking/plots', exist_ok=True)
        out = f"../2D-Tracking/plots/linear_single_trajectory_{traj_idx}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.show()


def print_trajectory_stats(states, measurements, dt=0.1):
    """
    Print statistics about the trajectories
    """
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    
    for traj_idx in range(states.shape[0]):
        print(f"\nTrajectory {traj_idx + 1}:")
        print("-" * 30)
        
        # Extract data
        x_pos = states[traj_idx, :, 0]
        y_pos = states[traj_idx, :, 1]
        x_vel = states[traj_idx, :, 2]
        y_vel = states[traj_idx, :, 3]
        x_meas = measurements[traj_idx, :, 0]
        y_meas = measurements[traj_idx, :, 1]
        
        # Calculate statistics
        total_distance = np.sum(np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2))
        avg_speed = np.mean(np.sqrt(x_vel**2 + y_vel**2))
        max_speed = np.max(np.sqrt(x_vel**2 + y_vel**2))
        
        # Measurement noise statistics
        meas_noise_x = x_meas - x_pos
        meas_noise_y = y_meas - y_pos
        meas_noise_std = np.std(np.sqrt(meas_noise_x**2 + meas_noise_y**2))
        
        print(f"  Duration: {len(x_pos) * dt:.1f} seconds")
        print(f"  Total distance: {total_distance:.3f} units")
        print(f"  Average speed: {avg_speed:.3f} units/s")
        print(f"  Maximum speed: {max_speed:.3f} units/s")
        print(f"  Start position: ({x_pos[0]:.3f}, {y_pos[0]:.3f})")
        print(f"  End position: ({x_pos[-1]:.3f}, {y_pos[-1]:.3f})")
        print(f"  Measurement noise std: {meas_noise_std:.3f} units")
        
        # Process noise statistics (velocity changes)
        vel_changes = np.sqrt(np.diff(x_vel)**2 + np.diff(y_vel)**2)
        process_noise_std = np.std(vel_changes)
        print(f"  Process noise std: {process_noise_std:.3f} units/s")


def main():
    """
    Main function to load and visualize trajectories
    """
    # Configuration: set the trajectory index to plot (0-based)
    selected_traj_idx = 1  # <-- change this to choose which trajectory to plot

    # File paths
    states_file = '../2D-Tracking/Saved_Data/2D_noisy_states.h5'
    measurements_file = '../2D-Tracking/Saved_Data/2D_noisy_measurements.h5'
    
    # Check if files exist
    if not os.path.exists(states_file):
        print(f"Error: States file not found: {states_file}")
        print("Please run 2D_generate_noisy_data first to create the HDF5 files.")
        return
    
    if not os.path.exists(measurements_file):
        print(f"Error: Measurements file not found: {measurements_file}")
        print("Please run 2D_generate_noisy_data first to create the HDF5 files.")
        return
    
    print("Loading trajectory data...")
    
    try:
        # Determine total available trajectories and clamp index
        with h5py.File(states_file, 'r') as f:
            total_traj = f['states'].shape[0]
        tidx = max(0, min(selected_traj_idx, total_traj - 1))

        # Load only the selected trajectory
        with h5py.File(states_file, 'r') as f:
            s_one = f['states'][tidx:tidx+1, :, :]
        with h5py.File(measurements_file, 'r') as f:
            m_one = f['measurements'][tidx:tidx+1, :, :]

        print(f"Plotting single trajectory index {tidx} (of {total_traj})")
        plot_single_trajectory(s_one, m_one, traj_idx=0, save_plot=True)
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("Please ensure the HDF5 files are properly formatted.")


if __name__ == "__main__":
    main() 
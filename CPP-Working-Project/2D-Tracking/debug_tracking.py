#!/usr/bin/env python3
"""
Debug script to test tracking scenarios and verify process noise effects
"""

import yaml
import subprocess
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def modify_config(use_process_noise):
    """Modify the YAML config to set process noise option"""
    config_path = '../BO_Parameters.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['Data_Generation']['use_process_noise'] = use_process_noise
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config updated: use_process_noise = {use_process_noise}")

def compile_and_run():
    """Compile and run the C++ data generation"""
    # We're already in the 2D-Tracking directory
    
    # Compile
    compile_cmd = ['g++', '-std=c++17', '-O3', '-I/usr/include/eigen3', 
                   '-I/usr/include/hdf5/serial', '-lhdf5_cpp', '-lhdf5',
                   '-lyaml-cpp', 'tracking_gen_data.cpp', '-o', 'tracking_gen_data']
    
    print("Compiling...")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        return False
    
    # Run
    print("Running data generation...")
    result = subprocess.run(['../build/tracking_gen_data'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Execution failed:")
        print(result.stderr)
        return False
    
    print("Data generation completed successfully")
    return True

def analyze_trajectory(states, measurements, scenario_name):
    """Analyze a single trajectory"""
    print(f"\n{scenario_name} Analysis:")
    print("-" * 40)
    
    # Extract first trajectory
    traj_states = states[0, :, :]  # Shape: (N, 4)
    traj_meas = measurements[0, :, :]  # Shape: (N, 2)
    
    # Calculate trajectory statistics
    x_pos = traj_states[:, 0]
    y_pos = traj_states[:, 1]
    x_vel = traj_states[:, 2]
    y_vel = traj_states[:, 3]
    
    # Calculate acceleration (should be zero for constant velocity)
    dt = 0.1
    x_accel = np.diff(x_vel) / dt
    y_accel = np.diff(y_vel) / dt
    accel_magnitude = np.sqrt(x_accel**2 + y_accel**2)
    
    # Calculate measurement noise
    x_meas = traj_meas[:, 0]
    y_meas = traj_meas[:, 1]
    meas_error = np.sqrt((x_meas - x_pos)**2 + (y_meas - y_pos)**2)
    
    print(f"  Position range: x=[{x_pos.min():.6f}, {x_pos.max():.6f}], y=[{y_pos.min():.6f}, {y_pos.max():.6f}]")
    print(f"  Velocity range: vx=[{x_vel.min():.6f}, {x_vel.max():.6f}], vy=[{y_vel.min():.6f}, {y_vel.max():.6f}]")
    print(f"  Max acceleration: {accel_magnitude.max():.6f}")
    print(f"  Acceleration std: {accel_magnitude.std():.6f}")
    print(f"  Measurement error std: {meas_error.std():.6f}")
    print(f"  Total distance: {np.sum(np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2)):.6f}")
    
    return traj_states, traj_meas

def test_scenarios():
    """Test both scenarios and compare results"""
    print("Testing tracking scenarios...")
    
    # Test 1: With process noise
    print("\n" + "="*60)
    print("TESTING WITH PROCESS NOISE")
    print("="*60)
    modify_config(True)
    if not compile_and_run():
        return
    
    # Load data
    with h5py.File('Saved_Data/2D_noisy_states.h5', 'r') as f:
        states_with_noise = f['states'][:]
    with h5py.File('Saved_Data/2D_noisy_measurements.h5', 'r') as f:
        meas_with_noise = f['measurements'][:]
    
    traj_with_noise, meas_with_noise_traj = analyze_trajectory(states_with_noise, meas_with_noise, "WITH Process Noise")
    
    # Test 2: Without process noise
    print("\n" + "="*60)
    print("TESTING WITHOUT PROCESS NOISE")
    print("="*60)
    modify_config(False)
    if not compile_and_run():
        return
    
    # Load data
    with h5py.File('Saved_Data/2D_noisy_states.h5', 'r') as f:
        states_without_noise = f['states'][:]
    with h5py.File('Saved_Data/2D_noisy_measurements.h5', 'r') as f:
        meas_without_noise = f['measurements'][:]
    
    traj_without_noise, meas_without_noise_traj = analyze_trajectory(states_without_noise, meas_without_noise, "WITHOUT Process Noise")
    
    # Compare trajectories
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Check if trajectories are identical
    states_diff = np.abs(traj_with_noise - traj_without_noise)
    meas_diff = np.abs(meas_with_noise_traj - meas_without_noise_traj)
    
    print(f"Max state difference: {states_diff.max():.10f}")
    print(f"Max measurement difference: {meas_diff.max():.10f}")
    print(f"States identical: {np.allclose(traj_with_noise, traj_without_noise, atol=1e-10)}")
    print(f"Measurements identical: {np.allclose(meas_with_noise_traj, meas_without_noise_traj, atol=1e-10)}")
    
    # Plot comparison
    plot_comparison(traj_with_noise, meas_with_noise, traj_without_noise, meas_without_noise)

def plot_comparison(traj_with_noise, meas_with_noise, traj_without_noise, meas_without_noise):
    """Plot comparison of trajectories"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tracking Scenarios Comparison', fontsize=16)
    
    dt = 0.1
    time = np.arange(len(traj_with_noise)) * dt
    
    # Plot 1: Position trajectories
    axes[0, 0].plot(traj_with_noise[:, 0], traj_with_noise[:, 1], 'b-', linewidth=2, label='With Process Noise')
    axes[0, 0].scatter(meas_with_noise[:, 0], meas_with_noise[:, 1], c='blue', alpha=0.6, s=20)
    
    axes[0, 0].plot(traj_without_noise[:, 0], traj_without_noise[:, 1], 'r--', linewidth=2, label='Without Process Noise')
    axes[0, 0].scatter(meas_without_noise[:, 0], meas_without_noise[:, 1], c='red', alpha=0.6, s=20)
    
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Position Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Velocity comparison
    speed_with = np.sqrt(traj_with_noise[:, 2]**2 + traj_with_noise[:, 3]**2)
    speed_without = np.sqrt(traj_without_noise[:, 2]**2 + traj_without_noise[:, 3]**2)
    
    axes[0, 1].plot(time, speed_with, 'b-', linewidth=2, label='With Process Noise')
    axes[0, 1].plot(time, speed_without, 'r--', linewidth=2, label='Without Process Noise')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed')
    axes[0, 1].set_title('Speed vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Acceleration comparison
    accel_with = np.sqrt(np.diff(traj_with_noise[:, 2])**2 + np.diff(traj_with_noise[:, 3])**2) / dt
    accel_without = np.sqrt(np.diff(traj_without_noise[:, 2])**2 + np.diff(traj_without_noise[:, 3])**2) / dt
    
    axes[1, 0].plot(time[1:], accel_with, 'b-', linewidth=2, label='With Process Noise')
    axes[1, 0].plot(time[1:], accel_without, 'r--', linewidth=2, label='Without Process Noise')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Acceleration')
    axes[1, 0].set_title('Acceleration vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Measurement error comparison
    meas_error_with = np.sqrt((meas_with_noise[:, 0] - traj_with_noise[:, 0])**2 + 
                             (meas_with_noise[:, 1] - traj_with_noise[:, 1])**2)
    meas_error_without = np.sqrt((meas_without_noise[:, 0] - traj_without_noise[:, 0])**2 + 
                               (meas_without_noise[:, 1] - traj_without_noise[:, 1])**2)
    
    axes[1, 1].plot(time, meas_error_with, 'b-', linewidth=2, label='With Process Noise')
    axes[1, 1].plot(time, meas_error_without, 'r--', linewidth=2, label='Without Process Noise')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Measurement Error')
    axes[1, 1].set_title('Measurement Error vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_tracking_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_scenarios() 
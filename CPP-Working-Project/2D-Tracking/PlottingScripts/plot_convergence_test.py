#!/usr/bin/env python3
"""
CNIS Convergence Visualization Script
====================================

Plots the convergence of CNIS components (NIS mean, variance, log components) 
as a function of the number of runs used in the Monte Carlo simulation.

Usage:
    python plot_convergence_test.py [--linear] [--nonlinear] [--both]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Install with: pip install pyyaml")

def load_convergence_data(filepath):
    """Load convergence data from CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None
    
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} data points from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_convergence(data, title_prefix="", save_prefix="", method="", show_components=True):
    """Create convergence plots for CNIS components."""
    
    # Create method display string
    method_str = ""
    if method:
        if method.lower() == "nis3":
            method_str = " (Proposition 3)"
        elif method.lower() == "nis4":
            method_str = " (Proposition 4)"
        else:
            method_str = f" ({method.upper()})"
    
    # Figure 1: NIS Mean and Variance
    fig1 = plt.figure(figsize=(12, 5))
    gs1 = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Get DOF for reference lines (should be constant across all data points)
    dof = data['dof'].iloc[0]
    
    # NIS Mean
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(data['num_runs'], data['nis_mean'], 'b-', linewidth=2, label='NIS Mean')
    # Add theoretical reference line
    ax1.axhline(y=dof, color='k', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'Theoretical (DOF = {dof})')
    ax1.set_xlabel('Number of Runs')
    ax1.set_ylabel('NIS Mean')
    ax1.set_title(f'{title_prefix}NIS Mean Convergence{method_str}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # NIS Variance
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(data['num_runs'], data['nis_variance'], 'r-', linewidth=2, label='NIS Variance')
    # Add theoretical reference line
    ax2.axhline(y=2*dof, color='k', linestyle='--', linewidth=1.5, alpha=0.7, 
                label=f'Theoretical (2×DOF = {2*dof})')
    ax2.set_xlabel('Number of Runs')
    ax2.set_ylabel('NIS Variance')
    ax2.set_title(f'{title_prefix}NIS Variance Convergence{method_str}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig1.suptitle(f'{title_prefix}NIS Statistics Convergence{method_str}', fontsize=16, fontweight='bold')
    
    # Save the first plot
    if save_prefix:
        output_file1 = f"../2D-Tracking/Saved_Data/nis_convergence_{save_prefix}.png"
        fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
        print(f"NIS convergence plot saved to: {output_file1}")
    
    # Figure 2: CNIS
    fig2 = plt.figure(figsize=(12, 5))
    ax3 = fig2.add_subplot(111)
    ax3.plot(data['num_runs'], data['cnis'], 'g-', linewidth=3)
    ax3.set_xlabel('Number of Runs')
    ax3.set_ylabel('CNIS')
    ax3.set_title(f'{title_prefix}CNIS Convergence{method_str}')
    ax3.grid(True, alpha=0.3)
    
    fig2.suptitle(f'{title_prefix}CNIS Convergence Analysis{method_str}', fontsize=16, fontweight='bold')
    
    # Save the second plot
    if save_prefix:
        output_file2 = f"../2D-Tracking/Saved_Data/cnis_convergence_{save_prefix}.png"
        fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"CNIS convergence plot saved to: {output_file2}")

def plot_comparison(linear_data, nonlinear_data, linear_method="", nonlinear_method=""):
    """Plot comparison between linear and nonlinear convergence."""
    
    # Create method display strings
    linear_method_str = ""
    nonlinear_method_str = ""
    if linear_method:
        if linear_method.lower() == "nis3":
            linear_method_str = " (Prop 3)"
        elif linear_method.lower() == "nis4":
            linear_method_str = " (Prop 4)"
        else:
            linear_method_str = f" ({linear_method.upper()})"
    
    if nonlinear_method:
        if nonlinear_method.lower() == "nis3":
            nonlinear_method_str = " (Prop 3)"
        elif nonlinear_method.lower() == "nis4":
            nonlinear_method_str = " (Prop 4)"
        else:
            nonlinear_method_str = f" ({nonlinear_method.upper()})"
    
    # Figure 1: NIS Mean and Variance Comparison
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('Linear vs Nonlinear NIS Statistics Comparison', fontsize=16, fontweight='bold')
    
    # Get DOF for reference lines
    linear_dof = linear_data['dof'].iloc[0]
    nonlinear_dof = nonlinear_data['dof'].iloc[0]
    
    # NIS Mean comparison
    axes1[0].plot(linear_data['num_runs'], linear_data['nis_mean'], 
                   'b-', linewidth=2, label=f'Linear{linear_method_str}')
    axes1[0].plot(nonlinear_data['num_runs'], nonlinear_data['nis_mean'], 
                   'r-', linewidth=2, label=f'Nonlinear{nonlinear_method_str}')
    # Add theoretical reference lines
    axes1[0].axhline(y=linear_dof, color='b', linestyle='--', linewidth=1.5, alpha=0.5, 
                     label=f'Linear Theory (DOF = {linear_dof})')
    axes1[0].axhline(y=nonlinear_dof, color='r', linestyle='--', linewidth=1.5, alpha=0.5, 
                     label=f'Nonlinear Theory (DOF = {nonlinear_dof})')
    axes1[0].set_xlabel('Number of Runs')
    axes1[0].set_ylabel('NIS Mean')
    axes1[0].set_title('NIS Mean Convergence')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # NIS Variance comparison
    axes1[1].plot(linear_data['num_runs'], linear_data['nis_variance'], 
                   'b-', linewidth=2, label=f'Linear{linear_method_str}')
    axes1[1].plot(nonlinear_data['num_runs'], nonlinear_data['nis_variance'], 
                   'r-', linewidth=2, label=f'Nonlinear{nonlinear_method_str}')
    # Add theoretical reference lines
    axes1[1].axhline(y=2*linear_dof, color='b', linestyle='--', linewidth=1.5, alpha=0.5, 
                     label=f'Linear Theory (2×DOF = {2*linear_dof})')
    axes1[1].axhline(y=2*nonlinear_dof, color='r', linestyle='--', linewidth=1.5, alpha=0.5, 
                     label=f'Nonlinear Theory (2×DOF = {2*nonlinear_dof})')
    axes1[1].set_xlabel('Number of Runs')
    axes1[1].set_ylabel('NIS Variance')
    axes1[1].set_title('NIS Variance Convergence')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save NIS comparison plot
    output_file1 = "../2D-Tracking/Saved_Data/nis_convergence_comparison.png"
    fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"NIS comparison plot saved to: {output_file1}")
    
    # Figure 2: CNIS Comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
    fig2.suptitle('Linear vs Nonlinear CNIS Comparison', fontsize=16, fontweight='bold')
    
    # CNIS comparison
    ax2.plot(linear_data['num_runs'], linear_data['cnis'], 
                   'b-', linewidth=3, label=f'Linear{linear_method_str}')
    ax2.plot(nonlinear_data['num_runs'], nonlinear_data['cnis'], 
                   'r-', linewidth=3, label=f'Nonlinear{nonlinear_method_str}')
    ax2.set_xlabel('Number of Runs')
    ax2.set_ylabel('CNIS')
    ax2.set_title('CNIS Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save CNIS comparison plot
    output_file2 = "../2D-Tracking/Saved_Data/cnis_convergence_comparison.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"CNIS comparison plot saved to: {output_file2}")

def get_method_from_yaml(scenario_type):
    """Read the actual method used from the YAML configuration file."""
    try:
        if scenario_type == "linear":
            config_file = "../scenario_linear.yaml"
        elif scenario_type == "nonlinear":
            config_file = "../scenario_nonlinear.yaml"
        else:
            return ""
        
        if not YAML_AVAILABLE:
            print(f"Warning: YAML not available, cannot read method from {config_file}")
            return ""
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Read the method from bayesopt.consistency_method
        method = config.get("bayesopt", {}).get("consistency_method", "unknown")
        print(f"Read method '{method}' from {config_file}")
        return method
        
    except Exception as e:
        print(f"Warning: Could not read method from YAML file: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description='Plot CNIS convergence test results')
    parser.add_argument('--linear', action='store_true', help='Plot linear convergence')
    parser.add_argument('--nonlinear', action='store_true', help='Plot nonlinear convergence')
    parser.add_argument('--both', action='store_true', help='Plot both and comparison')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    # Default to both if no specific option given
    if not (args.linear or args.nonlinear or args.both):
        args.both = True
    
    # File paths
    linear_file = "../2D-Tracking/Saved_Data/cnis_convergence_linear.csv"
    nonlinear_file = "../2D-Tracking/Saved_Data/cnis_convergence_nonlinear.csv"
    
    linear_data = None
    nonlinear_data = None
    linear_method = ""
    nonlinear_method = ""
    
    # Load data
    if args.linear or args.both:
        linear_data = load_convergence_data(linear_file)
        if linear_data is not None:
            linear_method = get_method_from_yaml("linear")
            plot_convergence(linear_data, title_prefix="Linear ", save_prefix="linear", method=linear_method)
    
    if args.nonlinear or args.both:
        nonlinear_data = load_convergence_data(nonlinear_file)
        if nonlinear_data is not None:
            nonlinear_method = get_method_from_yaml("nonlinear")
            plot_convergence(nonlinear_data, title_prefix="Nonlinear ", save_prefix="nonlinear", method=nonlinear_method)
    
    # Plot comparison if both datasets are available
    if args.both and linear_data is not None and nonlinear_data is not None:
        plot_comparison(linear_data, nonlinear_data, linear_method, nonlinear_method)
    
    plt.show()
    
    print("Convergence analysis complete!")

if __name__ == "__main__":
    main() 
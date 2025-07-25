# Tuned Parameter Demonstration and NEES Analysis

This directory contains scripts to demonstrate the tuned parameters and analyze NEES validation results.

## Files Created

### 1. `demo_tuned_parameters.cpp`
- **Purpose**: Demonstrates the factor graph with optimally tuned parameters
- **Input**: Uses tuned parameters from `../BO_Parameters.yaml`
- **Output**: Saves demonstration results to `Saved_Data/2D_tuned_demo_results.h5`
- **Features**: 
  - Runs 5 sample trajectories with tuned Q and R matrices
  - Calculates position/velocity estimation errors
  - Saves all states, estimates, and measurements to HDF5

### 2. `plot_tuned_demo.py` 
- **Purpose**: Visualizes the tuned parameter demonstration results
- **Input**: Reads `Saved_Data/2D_tuned_demo_results.h5`
- **Output**: Creates multiple plots in `plots/` directory:
  - `trajectory_comparison.png`: True vs estimated trajectories
  - `velocity_comparison.png`: Velocity analysis
  - `summary_statistics.png`: Overall performance statistics
  - `trajectory_run_*.png`: Individual run comparisons

### 3. `plot_nees_analysis.py`
- **Purpose**: Analyzes NEES convergence and distribution from validation results
- **Input**: Reads NEES validation files from `Validate_QR` output
- **Output**: Creates NEES analysis plots:
  - `nees_convergence_*.png`: Mean/variance convergence with number of runs  
  - `nees_distribution_*.png`: Distribution analysis and Q-Q plots
  - `nees_summary_*.png`: Summary of validation results

## Usage Instructions

### Step 1: Build the demonstration executable
```bash
cd CPP-Working-Project/build
make demo_tuned_parameters
```

### Step 2: Run the tuned parameter demonstration
```bash
cd CPP-Working-Project/build
./demo_tuned_parameters
```

This will:
- Load tuned parameters: q≈0.935, R≈1.301, CNEES≈0.0036
- Run factor graph optimization on 5 sample trajectories  
- Save results to HDF5 format
- Display performance statistics

### Step 3: Generate trajectory visualization plots
```bash
cd CPP-Working-Project/2D-Tracking
python plot_tuned_demo.py
```

### Step 4: Generate NEES analysis plots (requires validation data)
```bash
# First run validation (if not already done)
cd CPP-Working-Project/build
./Validate_QR  # or ./Validate_QR --use-existing-data

# Then create NEES plots
cd CPP-Working-Project/2D-Tracking
python plot_nees_analysis.py
```

## What the Plots Show

### Trajectory Plots
- **Green line**: True trajectory
- **Blue dashed**: Estimated trajectory  
- **Red dots**: Noisy measurements
- **Error plots**: Position/velocity errors over time

### NEES Analysis Plots
- **Combined convergence plot**: NEES mean and variance on same graph (0 to max variance + margin) showing convergence as Monte Carlo runs increase
- **Detailed mean convergence**: Zoomed view of mean convergence with confidence bands
- **Individual NEES scatter**: All NEES values plotted vs run number with theoretical references
- **Distribution analysis**: Histogram and Q-Q plots showing if NEES follows expected chi-squared distribution
- **Summary**: Key statistics and quality assessment

## Expected Results

With properly tuned parameters, you should see:
- **Low estimation errors**: Mean position error < 0.1 units
- **NEES convergence**: Sample mean/variance approach theoretical values (80, 160)
- **Good trajectory tracking**: Estimates closely follow true trajectories
- **Chi-squared consistency**: NEES values follow expected distribution

## Troubleshooting

- **File not found errors**: Make sure to run `demo_tuned_parameters` before plotting
- **Import errors**: Install required Python packages: `pip install h5py numpy matplotlib scipy`
- **Build errors**: Check that all dependencies (Eigen, g2o, HDF5, yaml-cpp) are installed 
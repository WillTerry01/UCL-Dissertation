# Automatic Tuning of Factor Graph-Based Estimation Using Bayesian Optimization

**Author:** Will Terry  
**Institution:** University College London (UCL)  
**Publication:** *To be determined*

---

## Overview

This repository implements a framework for **automatically tuning factor graph-based state estimation systems** using **Bayesian Optimization (BO)**. The primary focus is on a **2D GPS tracking problem** where we optimize noise covariance parameters (process noise `Q` and measurement noise `R`) to achieve optimal filter consistency.

### Key Features

- **Factor Graph Optimization**: Leverages the g2o library for efficient graph-based SLAM and state estimation
- **Bayesian Optimization**: Uses BayesOpt library to intelligently search the parameter space
- **Consistency Metrics**: Implements CNIS (Consistency-based Normalized Innovation Squared) and CNEES (Consistency-based Normalized Estimation Error Squared) for filter tuning
- **Dual Motion Models**: Supports both linear (constant velocity) and nonlinear (constant turn-rate) motion models
- **Comprehensive Pipeline**: From data generation to parameter optimization and validation

### Current Application

The current implementation focuses on a **2D GPS estimation problem** with:
- Linear motion model (constant velocity)
- Nonlinear motion model (constant turn-rate)
- GPS position measurements
- Automatic tuning of process and measurement noise parameters

### Future Work

The next phase will extend this framework to a **2D monocular camera tracking system** using bearing and range measurements to landmarks, demonstrating the generalizability of the approach.

---

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Version Information](#version-information)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- **CMake** (≥ 3.10)
- **C++ Compiler** with C++17 support (GCC, Clang, or MSVC)
- **Eigen3** (Linear algebra library)
- **HDF5** (Data storage)
- **Boost** (Required by BayesOpt)
- **OpenMP** (Optional, for parallel execution)

### System-Specific Installation

#### macOS (using Homebrew)

```bash
brew install cmake eigen hdf5 boost libomp
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install cmake libeigen3-dev libhdf5-dev libboost-all-dev libomp-dev
```

### Building the Project

**Important:** The build must be performed in a specific location within the repository structure.

1. **Clone the repository:**

```bash
git clone https://github.com/WillTerry01/UCL-Dissertation.git
cd UCL-Dissertation
```

2. **Build BayesOpt library** (included in repository):

The BayesOpt library is included in the `bayesopt/` directory and will be built automatically when you build the main project. The library is statically linked.

3. **Handle g2o dependency:**

The g2o library is used for factor graph optimization. While a `g2o/` directory exists in the repository, **you should install g2o system-wide** for easier use:

**macOS:**
```bash
brew install g2o
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libg2o-dev
```

Alternatively, you can build g2o from source and install it locally. The CMakeLists.txt currently points to a local installation path that you'll need to update:

```cmake
# In CPP-Working-Project/CMakeLists.txt, line 22
# Update this path to your g2o installation:
set(g2o_DIR "/path/to/your/g2o/installation/lib/cmake/g2o")
```

**Note:** The g2o submodule is primarily used for calculating CNEES values. For most use cases focusing on CNIS metrics, the standard g2o library installation is sufficient.

4. **Build the project:**

Navigate to the `CPP-Working-Project` directory and build:

```bash
cd CPP-Working-Project
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

All executables will be generated in the `build/` directory.

---

## Dependencies

### Required Libraries

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **CMake** | ≥ 3.10 | Build system | System package manager |
| **Eigen3** | ≥ 3.3 | Linear algebra | System package manager |
| **HDF5** | Latest | Data I/O | System package manager |
| **Boost** | ≥ 1.65 | BayesOpt dependency | System package manager |
| **g2o** | Latest | Factor graph optimization | System package manager or build from source |
| **OpenMP** | Latest | Parallelization (optional) | Usually included with compiler |

### Included Libraries

| Library | Location | Purpose |
|---------|----------|---------|
| **BayesOpt** | `bayesopt/` | Bayesian optimization engine |
| **yaml-cpp** | Fetched by CMake | YAML configuration parsing |

---

## Project Structure

```
UCL-Dissertation/
├── README.md                          # This file
├── bayesopt/                          # BayesOpt library (included)
├── g2o/                               # g2o placeholder (install system-wide instead)
├── CPP-Working-Project/
│   ├── CMakeLists.txt                # Main build configuration
│   ├── scenario_linear.yaml          # Configuration for linear motion model
│   ├── scenario_nonlinear.yaml       # Configuration for nonlinear motion model
│   └── 2D-Tracking/                  # Main source code directory
│       ├── fg_class_tracking.h/cpp   # Factor graph implementation
│       ├── 2D_h5_loader.h            # HDF5 data loading utilities
│       ├── tracking_gen_data_*.cpp   # Data generation programs
│       ├── BO_Tracking_Test_*.cpp    # Bayesian optimization programs
│       ├── GridSearch_Tracking_*.cpp # Grid search programs
│       ├── Evaluate_MSE_Points_*.cpp # MSE evaluation programs
│       ├── collect_nis_*.cpp         # NIS collection utilities
│       ├── convergence_test_*.cpp    # Convergence testing
│       ├── PlottingScripts/          # Python visualization scripts
│       └── Saved_Data/               # Output directory (generated)
└── build/                            # Build directory (generated)
```

---

## Quick Start

### 1. Generate Synthetic Data

First, generate synthetic trajectory data with noisy measurements:

```bash
cd CPP-Working-Project/build

# For linear motion model
./tracking_gen_data_linear

# For nonlinear motion model
./tracking_gen_data_nonlinear
```

This reads configuration from `scenario_linear.yaml` or `scenario_nonlinear.yaml` and generates HDF5 files in `2D-Tracking/Saved_Data/`.

### 2. Run Bayesian Optimization

Optimize the filter parameters using Bayesian Optimization:

```bash
# For linear model
./BO_Tracking_Test_linear

# For nonlinear model
./BO_Tracking_Test_Nonlinear
```

The optimization will search for optimal `Q` and `R` parameters that minimize the chosen consistency metric (CNIS or CNEES).

### 3. Evaluate Results

Evaluate the performance at specific parameter values:

```bash
# For linear model
./Evaluate_MSE_Points_Linear

# For nonlinear model
./Evaluate_MSE_Points_Nonlinear
```

---

## Usage

### Pipeline Workflow

The typical workflow for using this framework is:

1. **Configure Scenario**: Edit `scenario_linear.yaml` or `scenario_nonlinear.yaml` to set:
   - Trajectory parameters (length, initial conditions)
   - Noise parameters (for data generation)
   - Bayesian optimization settings
   - Grid search parameters (if using grid search)

2. **Generate Data**: Run the data generation executable to create synthetic trajectories and measurements

3. **Parameter Optimization**: Choose one of the following approaches:
   - **Bayesian Optimization** (recommended): Efficient search using BO
   - **Grid Search**: Exhaustive search over a parameter grid
   - **Point Evaluation**: Test specific parameter combinations

4. **Analyze Results**: Use the plotting scripts in `2D-Tracking/PlottingScripts/` to visualize results

### Configuration Files

All executables read their configuration from YAML files. The key sections are:

- **`Data_Generation`**: Controls synthetic data generation
- **`bayesopt`**: Bayesian optimization parameters
- **`grid_search`**: Grid search parameters
- **`parameters`**: Parameter bounds for optimization
- **`optimizer`**: Factor graph optimizer settings

Example configuration snippet:

```yaml
Data_Generation:
  trajectory_length: 100
  dt: 1.0
  q: 1.0                    # True process noise
  meas_noise_var: 2.0       # True measurement noise
  num_graphs: 10000         # Number of Monte Carlo runs

bayesopt:
  consistency_method: nis4  # CNIS metric variant
  n_iterations: 500         # BO iterations
  n_init_samples: 100       # Initial random samples

parameters:
  - name: q
    lower_bound: 0.001
    upper_bound: 2.0
  - name: R
    lower_bound: 0.001
    upper_bound: 4.0
```

### Available Executables

All executables should be run from the `CPP-Working-Project/build/` directory:

#### Data Generation
- `tracking_gen_data_linear` - Generate linear motion data
- `tracking_gen_data_nonlinear` - Generate nonlinear motion data

#### Bayesian Optimization
- `BO_Tracking_Test_linear` - BO for linear model
- `BO_Tracking_Test_Nonlinear` - BO for nonlinear model

#### Grid Search (CNIS/CNEES)
- `GridSearch_Tracking_linear` - Grid search using consistency metrics (linear)
- `GridSearch_Tracking_Nonlinear` - Grid search using consistency metrics (nonlinear)

#### Grid Search (MSE)
- `GridSearch_Tracking_Linear_MSE` - Grid search using MSE metric (linear)
- `GridSearch_Tracking_Nonlinear_MSE` - Grid search using MSE metric (nonlinear)

#### Evaluation
- `Evaluate_MSE_Points_Linear` - Evaluate MSE at specific points (linear)
- `Evaluate_MSE_Points_Nonlinear` - Evaluate MSE at specific points (nonlinear)

#### Analysis Tools
- `collect_nis_linear` - Collect NIS statistics (linear)
- `collect_nis_nonlinear` - Collect NIS statistics (nonlinear)
- `convergence_test_linear` - Test optimization convergence (linear)
- `convergence_test_nonlinear` - Test optimization convergence (nonlinear)
- `CrossSection_Tracking_Nonlinear` - Parameter landscape analysis

---

## Configuration

### Modifying Scenarios

To customize the experiments, edit the YAML configuration files:

- **`CPP-Working-Project/scenario_linear.yaml`** - Linear motion model settings
- **`CPP-Working-Project/scenario_nonlinear.yaml`** - Nonlinear motion model settings

### Key Configuration Parameters

**Data Generation:**
- `trajectory_length`: Number of time steps
- `dt`: Time step size (or use `dt_pieces` for variable time steps)
- `num_graphs`: Number of Monte Carlo runs
- `q`: True process noise intensity
- `meas_noise_var`: True measurement noise variance

**Bayesian Optimization:**
- `consistency_method`: Metric to optimize (`nis3`, `nis4`, etc.)
- `n_iterations`: Number of BO iterations
- `n_init_samples`: Initial random samples before BO starts
- `surr_name`: Surrogate model type
- `crit_name`: Acquisition function

**Parameter Bounds:**
- Define search space for `q` and `R` parameters

---

## Version Information

### Tested Environment

This project has been developed and tested with the following versions:

- **CMake**: 3.10+
- **C++ Standard**: C++17
- **Eigen**: 3.3+
- **HDF5**: 1.10+
- **Boost**: 1.65+
- **g2o**: Latest stable release
- **yaml-cpp**: 0.8.0 (fetched automatically)

**Note:** While the project may work with other versions, these are the versions used during development. If you encounter issues with different versions, please report them.

### Platform Support

- **macOS**: Expected to work (dependencies available)
- **Linux (Ubuntu/Debian)**: Fully Tested
- **Windows**: Not tested (may require modifications)

---

## Output Files

All output files are saved to `CPP-Working-Project/2D-Tracking/Saved_Data/`:

- **Data files**: `2D_noisy_states.h5`, `2D_noisy_measurements.h5`, etc.
- **Grid search results**: `2D_gridsearch_trials_*.h5`
- **MSE evaluations**: `2D_*_mse_point_evals.h5` and `.csv`
- **NIS collections**: `2D_nis_*.h5`

Visualization outputs can be found in `CPP-Working-Project/2D-Tracking/plots/` (generated by plotting scripts).

---

## Troubleshooting

### Common Issues

**1. g2o not found during CMake configuration**

Update the `g2o_DIR` path in `CPP-Working-Project/CMakeLists.txt` (line 22) to point to your g2o installation:

```cmake
set(g2o_DIR "/path/to/g2o/lib/cmake/g2o")
```

Or install g2o system-wide using your package manager.

**2. HDF5 library errors**

Ensure HDF5 is installed with C++ support:
```bash
# macOS
brew install hdf5

# Ubuntu/Debian
sudo apt-get install libhdf5-dev
```

**3. OpenMP not found**

OpenMP is optional but recommended for parallel execution. Install it:
```bash
# macOS
brew install libomp

# Ubuntu/Debian (usually included with GCC)
sudo apt-get install libomp-dev
```

---

## Contributing

This is a research project. For questions or collaboration inquiries, please contact the author.

---

## License

*License information to be added*

---

## Citation

If you use this work in your research, please cite:

```
@misc{terry2025autotuning,
  author = {Terry, Will},
  title = {Automatic Tuning of Factor Graph-Based Estimation Using Bayesian Optimization},
  year = {2025},
  institution = {University College London},
  note = {Publication pending}
}
```

---

## Acknowledgments

This project uses the following open-source libraries:
- [g2o](https://github.com/RainerKuemmerle/g2o) - General Graph Optimization
- [BayesOpt](https://github.com/rmcantin/bayesopt) - Bayesian Optimization Library
- [Eigen](https://eigen.tuxfamily.org/) - Linear Algebra Library
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) - High-Performance Data Management
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) - YAML Parser

# 2D-Tracking programs: what they do and how to run them

> WARNING: This was a AI generated README file to be used as a place holder, a full version is on the way

This project uses CMake. Build once from the repo root, then run the executables from the `build/` directory.

```bash
mkdir -p build && cd build
cmake ..
make -j
```

Before running, configure the scenario files:
- `CPP-Working-Project/scenario_linear.yaml`
- `CPP-Working-Project/scenario_nonlinear.yaml`

Both support variable time steps via `Data_Generation.dt_pieces`. Outputs are written to `CPP-Working-Project/2D-Tracking/Saved_Data/`.

## Data generation
- `2D-Tracking/tracking_gen_data_linear.cpp`
  - Generates linear (constant-velocity) trajectories + noisy measurements
  - Reads: `../scenario_linear.yaml`
  - Writes: `Saved_Data/2D_noisy_states.h5`, `Saved_Data/2D_noisy_measurements.h5`
  - Run: `./tracking_gen_data_linear`

- `2D-Tracking/tracking_gen_data_nonlinear.cpp`
  - Generates nonlinear (constant turn-rate) trajectories + Cartesian measurements
  - Reads: `../scenario_nonlinear.yaml`
  - Writes: `Saved_Data/2D_nonlinear_states.h5`, `Saved_Data/2D_nonlinear_measurements.h5`
  - Run: `./tracking_gen_data_nonlinear`

## Core factor-graph code
- `2D-Tracking/fg_class_tracking.h, .cpp`
  - Factor-graph model (vertices, edges, optimizers), linear and CT (nonlinear) motion, GPS measurements
- `2D-Tracking/2D_h5_loader.h`
  - Helpers to load HDF5 states/measurements

## Grid search (consistency metrics, CNIS/CNEES)
- `2D-Tracking/GridSearch_Tracking_linear.cpp`
  - Dense grid over Q (process noise intensity) and R (measurement noise variance)
  - Uses `bayesopt.consistency_method` (e.g., `nis3`, `nis4`) from YAML; honors `dt_pieces`
  - Reads linear H5 files; writes an H5 of tried points and objectives
  - Run: `./GridSearch_Tracking_linear`

- `2D-Tracking/GridSearch_Tracking_Nonlinear.cpp`
  - Same as above for nonlinear data/model (CT motion)
  - Run: `./GridSearch_Tracking_Nonlinear`

## Grid search (average position MSE)
- `2D-Tracking/GridSearch_Tracking_Linear_MSE.cpp`
  - Grid over Q,R; runs optimization; objective is average position MSE across runs
  - Prefers `mse_grid_search` block in YAML (falls back to `grid_search`)
  - Writes `..._linear_trials_mse.h5`
  - Run: `./GridSearch_Tracking_Linear_MSE`

- `2D-Tracking/GridSearch_Tracking_Nonlinear_MSE.cpp`
  - Same as above for nonlinear data/model
  - Writes `..._nonlinear_trials_mse.h5`
  - Run: `./GridSearch_Tracking_Nonlinear_MSE`

## Evaluate MSE at specific points
- `2D-Tracking/Evaluate_MSE_Points_Linear.cpp`
  - Evaluates average position MSE at specific (Q,R) pairs
  - Reads pairs from `mse_eval.tests` in `scenario_linear.yaml` (fallback to `nis_eval.tests`)
  - Writes `Saved_Data/2D_linear_mse_point_evals.h5` and `.csv`
  - Run: `./Evaluate_MSE_Points_Linear`

- `2D-Tracking/Evaluate_MSE_Points_Nonlinear.cpp`
  - Same for nonlinear data/model
  - Writes `Saved_Data/2D_mse_point_evals.h5` and `.csv`
  - Run: `./Evaluate_MSE_Points_Nonlinear`

## Bayesian Optimization (CNIS/CNEES-driven)
- `2D-Tracking/BO_Tracking_Test_linear.cpp`
  - Runs BO over Q,R using the chosen consistency metric; honors variable `dt`
  - Reads: linear H5 + `scenario_linear.yaml`
  - Run: `./BO_Tracking_Test_linear`

- `2D-Tracking/BO_Tracking_Test_Nonlinear.cpp`
  - Same for nonlinear data/model
  - Run: `./BO_Tracking_Test_Nonlinear`

## NIS collection utilities
- `2D-Tracking/collect_nis_linear.cpp`
  - Computes and aggregates NIS values over runs (linear)
  - Run: `./collect_nis_linear`

- `2D-Tracking/collect_nis_nonlinear.cpp`
  - Same for nonlinear
  - Run: `./collect_nis_nonlinear`

## Convergence tests
- `2D-Tracking/convergence_test_linear.cpp`
  - Stress-test optimization convergence across runs (linear)
  - Run: `./convergence_test_linear`

- `2D-Tracking/convergence_test_nonlinear.cpp`
  - Same for nonlinear
  - Run: `./convergence_test_nonlinear`

## Misc/advanced
- `2D-Tracking/CrossSection_Tracking_Nonlinear.cpp`
  - Slices the objective (e.g., CNIS/MSE) along one parameter while holding the other fixed for landscape diagnostics
  - Run: `./CrossSection_Tracking_Nonlinear`

- `2D-Tracking/run_nonlinear_bo_5x.sh`
  - Convenience script to run multiple nonlinear BO trials; edit and execute as needed
  - Run: `bash 2D-Tracking/run_nonlinear_bo_5x.sh`

## Typical workflow
1) Configure `scenario_linear.yaml` and/or `scenario_nonlinear.yaml` (including `dt_pieces` if varying time steps)
2) Generate data:
   - Linear: `./tracking_gen_data_linear`
   - Nonlinear: `./tracking_gen_data_nonlinear`
3) Explore parameters via grid search or BO:
   - CNIS/CNEES grid: `./GridSearch_Tracking_linear` or `./GridSearch_Tracking_Nonlinear`
   - MSE grid: `./GridSearch_Tracking_Linear_MSE` or `./GridSearch_Tracking_Nonlinear_MSE`
   - BO: `./BO_Tracking_Test_linear` or `./BO_Tracking_Test_Nonlinear`
4) Evaluate specific points (optional):
   - Linear: `./Evaluate_MSE_Points_Linear`
   - Nonlinear: `./Evaluate_MSE_Points_Nonlinear`

Notes:
- Outputs are saved under `2D-Tracking/Saved_Data/`. The `2D-Tracking/plots/` folder is for images produced by separate plotting scripts (not covered here).
- All tools honor per-step `dt` schedules derived from `dt_pieces` where applicable.

/*
 * 2D Tracking Data Generation
 * ===========================
 * 
 * PURPOSE:
 * This script generates synthetic 2D tracking data for testing factor graph
 * optimization algorithms. It creates realistic trajectories with process noise
 * and corresponding noisy measurements for Monte Carlo analysis.
 * 
 * WORKFLOW:
 * 1. Load parameters from YAML config file (initial conditions, noise levels, etc.)
 * 2. Generate clean initial state from YAML parameters [x, y, vx, vy]
 * 3. For each Monte Carlo run:
 *    - Create trajectory using constant velocity model: x_{k+1} = F*x_k + process_noise
 *    - Apply process noise (if enabled) using proper multivariate sampling from Q matrix
 *    - Generate noisy position measurements: z_k = H*x_k + measurement_noise
 * 
 * OUTPUTS (HDF5 format):
 * 
 * File: "2D_noisy_states.h5"
 * - Dataset: "states" 
 * - Dimensions: [num_graphs, trajectory_length, 4]
 * - Content: Full state trajectories [x, y, vx, vy] WITH process noise applied
 * - Purpose: Represents the "ground truth reality" that the factor graph tries to estimate
 * 
 * File: "2D_noisy_measurements.h5"
 * - Dataset: "measurements"
 * - Dimensions: [num_graphs, trajectory_length, 2] 
 * - Content: Position observations [x_obs, y_obs] WITH measurement noise applied
 * - Purpose: The noisy observations that the factor graph uses for estimation
 * 
 * FACTOR GRAPH USAGE:
 * - The states file provides the "true_states" parameter (noisy reality to compare against)
 * - The measurements file provides the "measurements" parameter (observations for optimization)
 * - NIS3: Initialize exactly at noisy states, no optimization (tests graph structure)
 * - NIS4: Initialize at zero, optimize using measurements (tests convergence robustness)
 * 
 * This approach follows the validated MATLAB research implementation and avoids 
 * double application of process noise.
 */

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include "H5Cpp.h"
#include <yaml-cpp/yaml.h>

// Function declaration for control input generation
Eigen::Vector2d generateControlInput(int time_step, double dt);

static Eigen::Matrix4d buildQ(double q_intensity, double dt) {
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double V0 = q_intensity;
    double V1 = q_intensity;
    double min_variance = 1e-8;
    double dt_safe = std::max(dt, 1e-9);
    V0 = std::max(V0, min_variance / dt_safe);
    V1 = std::max(V1, min_variance / dt_safe);
    Q(0, 0) = dt3 / 3.0 * V0;
    Q(1, 1) = dt3 / 3.0 * V1;
    Q(2, 2) = dt * V0;
    Q(3, 3) = dt * V1;
    Q(0, 2) = dt2 / 2.0 * V0;
    Q(2, 0) = Q(0, 2);
    Q(1, 3) = dt2 / 2.0 * V1;
    Q(3, 1) = Q(1, 3);
    return Q;
}

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");

    // Parameters
    int N = config["Data_Generation"]["trajectory_length"].as<int>(); // Trajectory length
    int num_graphs = config["Data_Generation"]["num_graphs"].as<int>(); // Number of Monte Carlo samples
    double dt_default = config["Data_Generation"]["dt"].as<double>();
    Eigen::Vector2d pos(config["Data_Generation"]["pos"]["x"].as<double>(), config["Data_Generation"]["pos"]["y"].as<double>());
    Eigen::Vector2d vel(config["Data_Generation"]["vel"]["x"].as<double>(), config["Data_Generation"]["vel"]["y"].as<double>());
    unsigned int base_seed = config["Data_Generation"]["seed"].as<unsigned int>();
    
    // Build per-step dt schedule (length N-1). If not provided, use uniform dt.
    std::vector<double> dt_vec(std::max(0, N - 1), dt_default);
    if (config["Data_Generation"]["dt_pieces"]) {
        for (const auto& piece : config["Data_Generation"]["dt_pieces"]) {
            int from = piece["from"].as<int>();
            int to = piece["to"].as<int>();
            double dt_piece = piece["dt"].as<double>();
            from = std::max(0, from);
            to = std::min(N - 2, to);
            for (int k = from; k <= to; ++k) {
                dt_vec[k] = dt_piece;
            }
        }
    }
    
    // Noise parameters - V₀ and V₁ are the continuous-time white noise intensities for acceleration
    double V0 = config["Data_Generation"]["q"].as<double>();  // x-direction acceleration noise intensity
    double V1 = config["Data_Generation"]["q"].as<double>();  // y-direction acceleration noise intensity (same as V0 for now)
    double meas_noise_var = config["Data_Generation"]["meas_noise_var"].as<double>();  // This is actually measurement noise variance
    double meas_noise_std = sqrt(meas_noise_var);  // Actual standard deviation
    
    // NEW: Option to disable process noise for pure tracking scenarios
    bool use_process_noise = config["Data_Generation"]["use_process_noise"].as<bool>(true);  // Default to true
    if (!use_process_noise) {
        V0 = 0.0;
        V1 = 0.0;
        std::cout << "Process noise disabled - pure tracking scenario" << std::endl;
    } else {
        std::cout << "Process noise enabled - tracking with motion uncertainty" << std::endl;
    }

    std::cout << "Linear data gen: default dt = " << dt_default << ", variable schedule: "
              << (config["Data_Generation"]["dt_pieces"] ? "yes" : "no") << std::endl;

    // Construct the measurement noise covariance matrix R
    // R = [meas_noise_std^2    0              ]
    //     [0                   meas_noise_std^2]
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = meas_noise_var;  // x measurement variance
    R(1, 1) = meas_noise_var;  // y measurement variance

    Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) {
        std::cerr << "ERROR: R matrix is not positive semi-definite!" << std::endl;
        std::cerr << "R matrix:" << std::endl << R << std::endl;
        std::cerr << "meas_noise_var = " << meas_noise_var << ", meas_noise_std = " << meas_noise_std << std::endl;
        return -1;
    }

    // Prepare output arrays
    std::vector<double> states(num_graphs * N * 4);
    std::vector<double> measurements(num_graphs * N * 2);

    for (int run = 0; run < num_graphs; ++run) {
        // True trajectory
        std::vector<Eigen::Vector4d> true_states(N);
        true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        
        // Add process noise using per-step Q matrix
        std::mt19937 gen(base_seed + run); // Different seed for each run
        std::normal_distribution<> normal_dist(0.0, 1.0);
        std::vector<Eigen::Vector4d> noisy_states = true_states;
        
        for (int k = 1; k < N; ++k) {
            const double dt_k = dt_vec[k-1];
            // State transition matrix F_k (constant velocity model)
            Eigen::Matrix4d Fk = Eigen::Matrix4d::Identity();
            Fk(0, 2) = dt_k;
            Fk(1, 3) = dt_k;
            
            // Control input matrix B_k
            Eigen::Matrix<double, 4, 2> Bk;
            double dt2 = dt_k * dt_k;
            Bk << 0.5 * dt2, 0.0,
                  0.0, 0.5 * dt2,
                  dt_k, 0.0,
                  0.0, dt_k;
            
            // Generate control input (acceleration) - currently zero
            Eigen::Vector2d acceleration = generateControlInput(k, dt_k);
            
            // State equation: xₖ₊₁ = F_k xₖ + B_k uₖ + vₖ
            Eigen::Vector4d control_effect = Bk * acceleration;
            true_states[k] = Fk * true_states[k-1] + control_effect;
            
            // Apply process noise only if enabled
            if (use_process_noise) {
                Eigen::Matrix4d Qk = buildQ(V0, dt_k);
                Eigen::LLT<Eigen::Matrix4d> lltQk(Qk);
                if (lltQk.info() != Eigen::Success) {
                    std::cerr << "ERROR: Q(dt_k) is not positive semi-definite at step " << k << std::endl;
                    return -1;
                }
                Eigen::Matrix4d L = lltQk.matrixL();
                Eigen::Vector4d uncorrelated_noise;
                for (int i = 0; i < 4; ++i) {
                    uncorrelated_noise[i] = normal_dist(gen);
                }
                Eigen::Vector4d process_noise = L * uncorrelated_noise;
                true_states[k] += process_noise;
            }
            
            noisy_states[k] = true_states[k];
        }

        // Generate noisy measurements from noisy states using R matrix
        Eigen::Matrix2d L_R = lltOfR.matrixL();
        std::vector<Eigen::Vector2d> noisy_measurements(N);
        for (int k = 0; k < N; ++k) {
            Eigen::Vector2d uncorrelated_meas_noise;
            for (int i = 0; i < 2; ++i) {
                uncorrelated_meas_noise[i] = normal_dist(gen);
            }
            Eigen::Vector2d measurement_noise = L_R * uncorrelated_meas_noise;
            noisy_measurements[k] = noisy_states[k].head<2>() + measurement_noise;
        }

        // Store in output arrays
        for (int k = 0; k < N; ++k) {
            int state_idx = run * N * 4 + k * 4;
            states[state_idx + 0] = noisy_states[k][0];
            states[state_idx + 1] = noisy_states[k][1];
            states[state_idx + 2] = noisy_states[k][2];
            states[state_idx + 3] = noisy_states[k][3];
            int meas_idx = run * N * 2 + k * 2;
            measurements[meas_idx + 0] = noisy_measurements[k][0];
            measurements[meas_idx + 1] = noisy_measurements[k][1];
        }
    }

    // Save to HDF5
    const std::string states_h5 = "../2D-Tracking/Saved_Data/2D_noisy_states.h5";
    const std::string meas_h5 = "../2D-Tracking/Saved_Data/2D_noisy_measurements.h5";
    hsize_t states_dims[3] = {static_cast<hsize_t>(num_graphs), static_cast<hsize_t>(N), 4};
    hsize_t meas_dims[3] = {static_cast<hsize_t>(num_graphs), static_cast<hsize_t>(N), 2};

    // States
    H5::H5File states_file(states_h5, H5F_ACC_TRUNC);
    H5::DataSpace states_space(3, states_dims);
    H5::DataSet states_dataset = states_file.createDataSet("states", H5::PredType::NATIVE_DOUBLE, states_space);
    states_dataset.write(states.data(), H5::PredType::NATIVE_DOUBLE);

    // Measurements
    H5::H5File meas_file(meas_h5, H5F_ACC_TRUNC);
    H5::DataSpace meas_space(3, meas_dims);
    H5::DataSet meas_dataset = meas_file.createDataSet("measurements", H5::PredType::NATIVE_DOUBLE, meas_space);
    meas_dataset.write(measurements.data(), H5::PredType::NATIVE_DOUBLE);

    std::cout << "\nSaved all noisy states to " << states_h5 << " and all measurements to " << meas_h5 << std::endl;
    std::cout << "Using measurement model: zₖ = Hxₖ + wₖ" << std::endl;
    std::cout << "V0 = " << V0 << ", V1 = " << V1 << ", default dt = " << dt_default << std::endl;
    return 0;
}

// Control input generation function - currently returns zero for constant velocity
// TODO: Modify this function to generate non-zero acceleration for controlled motion
Eigen::Vector2d generateControlInput(int time_step, double dt) {
    // For now, return zero acceleration (constant velocity)
    // This can be modified later to generate:
    // - Sinusoidal acceleration: [A*sin(ω*t), A*cos(ω*t)]
    // - Step acceleration: [a_x, a_y] for t > t_switch
    // - Random acceleration: [N(0,σ²), N(0,σ²)]
    // - Nonlinear control laws
    return Eigen::Vector2d::Zero();
} 
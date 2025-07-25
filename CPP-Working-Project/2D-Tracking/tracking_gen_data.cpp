#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include "H5Cpp.h"
#include <yaml-cpp/yaml.h>

// Function declaration for control input generation
Eigen::Vector2d generateControlInput(int time_step, double dt);

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");

    // Parameters
    int N = config["Data_Generation"]["trajectory_length"].as<int>(); // Trajectory length
    int num_graphs = config["Data_Generation"]["num_graphs"].as<int>(); // Number of Monte Carlo samples
    double dt = config["Data_Generation"]["dt"].as<double>();
    Eigen::Vector2d pos(config["Data_Generation"]["pos"]["x"].as<double>(), config["Data_Generation"]["pos"]["y"].as<double>());
    Eigen::Vector2d vel(config["Data_Generation"]["vel"]["x"].as<double>(), config["Data_Generation"]["vel"]["y"].as<double>());
    unsigned int base_seed = config["Data_Generation"]["seed"].as<unsigned int>();
    
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

    // State transition matrix F (constant velocity model)
    // F = [1  0  dt  0]
    //     [0  1  0   dt]
    //     [0  0  1   0]
    //     [0  0  0   1]
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  // x position += x velocity * dt
    F(1, 3) = dt;  // y position += y velocity * dt

    // Control input matrix B (for future acceleration control)
    // B = [0.5 * dt^2  0        ]
    //     [0           0.5 * dt^2]
    //     [dt          0        ]
    //     [0           dt       ]
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;

    // Construct the process noise covariance matrix Q for 2D linear tracking
    // Q = [dt^3/3 * V₀    0           dt^2/2 * V₀    0        ]
    //     [0               dt^3/3 * V₁ 0               dt^2/2 * V₁]
    //     [dt^2/2 * V₀    0           dt * V₀         0        ]
    //     [0               dt^2/2 * V₁ 0               dt * V₁  ]
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt3 = dt2 * dt;
    
    // Position-position covariance (diagonal)
    Q(0, 0) = dt3 / 3.0 * V0;  // x position variance
    Q(1, 1) = dt3 / 3.0 * V1;  // y position variance
    
    // Velocity-velocity covariance (diagonal)
    Q(2, 2) = dt * V0;         // x velocity variance
    Q(3, 3) = dt * V1;         // y velocity variance
    
    // Position-velocity cross covariance
    Q(0, 2) = dt2 / 2.0 * V0;  // x position - x velocity covariance
    Q(2, 0) = Q(0, 2);         // symmetric
    Q(1, 3) = dt2 / 2.0 * V1;  // y position - y velocity covariance
    Q(3, 1) = Q(1, 3);         // symmetric

    // Construct the measurement noise covariance matrix R
    // R = [meas_noise_std^2    0              ]
    //     [0                   meas_noise_std^2]
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = meas_noise_var;  // x measurement variance
    R(1, 1) = meas_noise_var;  // y measurement variance

    // Validate that Q and R are positive semi-definite
    Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q);
    if (lltOfQ.info() != Eigen::Success) {
        std::cerr << "ERROR: Q matrix is not positive semi-definite!" << std::endl;
        std::cerr << "Q matrix:" << std::endl << Q << std::endl;
        std::cerr << "V0 = " << V0 << ", V1 = " << V1 << ", dt = " << dt << std::endl;
        return -1;
    }

    Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) {
        std::cerr << "ERROR: R matrix is not positive semi-definite!" << std::endl;
        std::cerr << "R matrix:" << std::endl << R << std::endl;
        std::cerr << "meas_noise_var = " << meas_noise_var << ", meas_noise_std = " << meas_noise_std << std::endl;
        return -1;
    }

    std::cout << "Q and R matrices are positive semi-definite ✓" << std::endl;
    std::cout << "State transition matrix F:" << std::endl << F << std::endl;
    std::cout << "Control input matrix B:" << std::endl << B << std::endl;
    std::cout << "Q matrix:" << std::endl << Q << std::endl;
    std::cout << "R matrix:" << std::endl << R << std::endl;
    
    // Debug: Show the Cholesky factor L when process noise is disabled
    if (!use_process_noise) {
        Eigen::Matrix4d L = lltOfQ.matrixL();
        std::cout << "Cholesky factor L (should be zero matrix):" << std::endl << L << std::endl;
    }

    // Prepare output arrays
    std::vector<double> states(num_graphs * N * 4);
    std::vector<double> measurements(num_graphs * N * 2);

    for (int run = 0; run < num_graphs; ++run) {
        // True trajectory
        std::vector<Eigen::Vector4d> true_states(N);
        true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        
        // Add process noise using Q matrix
        std::mt19937 gen(base_seed + run); // Different seed for each run
        std::vector<Eigen::Vector4d> noisy_states = true_states;
        
        // Generate correlated process noise using Cholesky decomposition
        Eigen::Matrix4d L = lltOfQ.matrixL();
        
        for (int k = 1; k < N; ++k) {
            // Generate control input (acceleration) - set to zero for constant velocity
            // TODO: Change this function to generate non-zero acceleration for controlled motion
            Eigen::Vector2d acceleration = generateControlInput(k, dt);  // Currently returns zero
            
            // State equation: xₖ₊₁ = Fxₖ + Buₖ + vₖ
            Eigen::Vector4d control_effect = B * acceleration;
            true_states[k] = F * true_states[k-1] + control_effect;
            
            // Apply process noise only if enabled
            if (use_process_noise) {
                // Generate uncorrelated standard normal noise
                Eigen::Vector4d uncorrelated_noise;
                std::normal_distribution<> normal_dist(0.0, 1.0);
                for (int i = 0; i < 4; ++i) {
                    uncorrelated_noise[i] = normal_dist(gen);
                }
                
                // Transform to correlated noise using Q = L*L^T
                Eigen::Vector4d process_noise = L * uncorrelated_noise;
                true_states[k] += process_noise;
            }
            
            // Copy to noisy states (for consistency with existing code)
            noisy_states[k] = true_states[k];
        }

        // Generate noisy measurements from noisy states using R matrix
        Eigen::Matrix2d L_R = lltOfR.matrixL();
        std::vector<Eigen::Vector2d> noisy_measurements(N);
        for (int k = 0; k < N; ++k) {
            // Generate uncorrelated standard normal measurement noise
            Eigen::Vector2d uncorrelated_meas_noise;
            std::normal_distribution<> normal_dist(0.0, 1.0);
            for (int i = 0; i < 2; ++i) {
                uncorrelated_meas_noise[i] = normal_dist(gen);
            }
            
            // Transform to correlated measurement noise using R = L_R*L_R^T
            Eigen::Vector2d measurement_noise = L_R * uncorrelated_meas_noise;
            
            // Add measurement noise to position measurements
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

    std::cout << "\nFinal system matrices used for all runs:" << std::endl;
    std::cout << "F matrix:" << std::endl << F << std::endl;
    std::cout << "B matrix:" << std::endl << B << std::endl;
    std::cout << "Q matrix:" << std::endl << Q << std::endl;
    std::cout << "R matrix:" << std::endl << R << std::endl;
    std::cout << "\nSaved all noisy states to " << states_h5 << " and all measurements to " << meas_h5 << std::endl;
    if (use_process_noise) {
        std::cout << "Using state-space model: xₖ₊₁ = Fxₖ + Buₖ + vₖ with uₖ = 0 (constant velocity + process noise)" << std::endl;
    } else {
        std::cout << "Using state-space model: xₖ₊₁ = Fxₖ + Buₖ (constant velocity, no process noise)" << std::endl;
    }
    std::cout << "Using measurement model: zₖ = Hxₖ + wₖ" << std::endl;
    std::cout << "V0 = " << V0 << ", V1 = " << V1 << ", dt = " << dt << std::endl;
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
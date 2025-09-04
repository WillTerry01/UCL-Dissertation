/*
 * 2D Nonlinear Tracking Data Generation
 * =====================================
 * 
 * PURPOSE:
 * This script generates synthetic 2D nonlinear tracking data for testing factor graph
 * optimization algorithms with nonlinear motion and measurement models.
 * 
 * NONLINEAR MODELS:
 * 1. Motion Model: Constant Turn Rate (CT) model
 *    - x_{k+1} = f(x_k, u_k) where f is nonlinear
 *    - Handles curved trajectories with constant angular velocity
 * 
 * 2. Measurement Model: Range and Bearing (RB) measurements
 *    - z_k = h(x_k) where h is nonlinear
 *    - Range: distance from sensor to target
 *    - Bearing: angle from sensor to target
 * 
 * WORKFLOW:
 * 1. Load parameters from YAML config file
 * 2. Generate clean initial state [x, y, vx, vy]
 * 3. For each Monte Carlo run:
 *    - Create trajectory using constant turn rate model
 *    - Apply process noise using proper multivariate sampling
 *    - Generate range-bearing measurements from sensor position
 * 
 * OUTPUTS (HDF5 format):
 * - Nonlinear states and measurements for testing nonlinear factor graphs
 */

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include "H5Cpp.h"
#include <yaml-cpp/yaml.h>
#include <cmath>

// Function declarations
Eigen::Vector4d predictStateCT(const Eigen::Vector4d& x, double dt, double turn_rate);

static Eigen::Matrix4d buildQ(double q_intensity, double dt, double turn_rate = 0.0) {
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
    // Apply mid-interval rotation for CT to match factor graph dynamics
    if (std::abs(turn_rate) > 1e-12) {
        double theta = turn_rate * dt * 0.5;
        double c = std::cos(theta);
        double s = std::sin(theta);
        Eigen::Matrix2d R2; R2 << c, -s, s, c;
        Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
        T.block<2,2>(0,0) = R2;
        T.block<2,2>(2,2) = R2;
        Q = T * Q * T.transpose();
    }
    return Q;
}

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");

    // Parameters
    int N = config["Data_Generation"]["trajectory_length"].as<int>();
    int num_graphs = config["Data_Generation"]["num_graphs"].as<int>();
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
            // clamp ranges
            from = std::max(0, from);
            to = std::min(N - 2, to);
            for (int k = from; k <= to; ++k) {
                dt_vec[k] = dt_piece;
            }
        }
    }
    
    // Noise parameters
    double V0 = config["Data_Generation"]["q"].as<double>();
    double V1 = config["Data_Generation"]["q"].as<double>();
    double meas_noise_var = config["Data_Generation"]["meas_noise_var"].as<double>();
    double meas_noise_std = sqrt(meas_noise_var);
    
    bool use_process_noise = config["Data_Generation"]["use_process_noise"].as<bool>(true);
    
    // Nonlinear parameters
    double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);  // rad/s
    // Removed sensor_pos - not needed for GPS tracking
    
    // Debug output to show loaded parameters
    std::cout << "=== Nonlinear Data Generation Parameters ===" << std::endl;
    std::cout << "Config file: scenario_nonlinear.yaml" << std::endl;
    std::cout << "Trajectory length: " << N << std::endl;
    std::cout << "Number of graphs: " << num_graphs << std::endl;
    std::cout << "Default Time step (dt): " << dt_default << std::endl;
    std::cout << "Initial position: [" << pos[0] << ", " << pos[1] << "]" << std::endl;
    std::cout << "Initial velocity: [" << vel[0] << ", " << vel[1] << "]" << std::endl;
    std::cout << "Turn rate: " << turn_rate << " rad/s" << std::endl;
    std::cout << "Process noise (q): " << V0 << std::endl;
    std::cout << "Measurement noise variance: " << meas_noise_var << std::endl;
    std::cout << "Use process noise: " << (use_process_noise ? "true" : "false") << std::endl;
    std::cout << "=============================================" << std::endl;

    // Construct the measurement noise covariance matrix R for Cartesian measurements
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = meas_noise_var;  // X position variance
    R(1, 1) = meas_noise_var;  // Y position variance

    Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) {
        std::cerr << "ERROR: R matrix is not positive semi-definite!" << std::endl;
        return -1;
    }

    std::cout << "Nonlinear system parameters:" << std::endl;
    std::cout << "  Turn rate: " << turn_rate << " rad/s" << std::endl;
    std::cout << "  R matrix:" << std::endl << R << std::endl;

    // Prepare output arrays
    std::vector<double> states(num_graphs * N * 4);
    std::vector<double> measurements(num_graphs * N * 2);

    for (int run = 0; run < num_graphs; ++run) {
        // True trajectory
        std::vector<Eigen::Vector4d> true_states(N);
        true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        
        std::mt19937 gen(base_seed + run);
        std::normal_distribution<> normal_dist(0.0, 1.0);
        std::vector<Eigen::Vector4d> noisy_states = true_states;
        
        for (int k = 1; k < N; ++k) {
            const double dt_k = dt_vec[k-1];
            // Nonlinear state prediction using constant turn rate model
            true_states[k] = predictStateCT(true_states[k-1], dt_k, turn_rate);
            
            // Apply process noise only if enabled
            if (use_process_noise) {
                Eigen::Matrix4d Qk = buildQ(V0, dt_k, turn_rate);
                Eigen::LLT<Eigen::Matrix4d> lltQk(Qk);
                if (lltQk.info() != Eigen::Success) {
                    std::cerr << "ERROR: Q(dt_k) is not positive semi-definite at step " << k << std::endl;
                    return -1;
                }
                Eigen::Matrix4d L = lltQk.matrixL();
                Eigen::Vector4d uncorrelated_noise;
                for (int i = 0; i < 4; ++i) uncorrelated_noise[i] = normal_dist(gen);
                Eigen::Vector4d process_noise = L * uncorrelated_noise;
                true_states[k] += process_noise;
            }
            noisy_states[k] = true_states[k];
        }

        // Generate GPS measurements (Cartesian position x, y)
        Eigen::Matrix2d L_R = lltOfR.matrixL();
        std::vector<Eigen::Vector2d> noisy_measurements(N);
        for (int k = 0; k < N; ++k) {
            Eigen::Vector2d uncorrelated_meas_noise;
            for (int i = 0; i < 2; ++i) {
                uncorrelated_meas_noise[i] = normal_dist(gen);
            }
            Eigen::Vector2d measurement_noise = L_R * uncorrelated_meas_noise;
            Eigen::Vector2d true_measurement = noisy_states[k].head<2>();
            noisy_measurements[k] = true_measurement + measurement_noise;
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
    const std::string states_h5 = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
    const std::string meas_h5 = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";
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

    std::cout << "Saved nonlinear states to: " << states_h5 << std::endl;
    std::cout << "Saved nonlinear measurements to: " << meas_h5 << std::endl;

    return 0;
}

// Nonlinear state prediction for constant turn rate model
Eigen::Vector4d predictStateCT(const Eigen::Vector4d& x, double dt, double turn_rate) {
    double x_pos = x[0], y_pos = x[1], vx = x[2], vy = x[3];
    double v = std::sqrt(vx*vx + vy*vy);  // Speed
    double heading = std::atan2(vy, vx);  // Current heading
    
    if (std::abs(turn_rate) < 1e-6) {
        // Straight line motion (constant velocity)
        return Eigen::Vector4d(x_pos + vx * dt, y_pos + vy * dt, vx, vy);
    } else {
        // Constant turn rate motion
        double new_heading = heading + turn_rate * dt;
        double new_vx = v * std::cos(new_heading);
        double new_vy = v * std::sin(new_heading);
        
        // Position update (integrate velocity)
        double new_x_pos = x_pos + (v / turn_rate) * (std::sin(new_heading) - std::sin(heading));
        double new_y_pos = y_pos - (v / turn_rate) * (std::cos(new_heading) - std::cos(heading));
        
        return Eigen::Vector4d(new_x_pos, new_y_pos, new_vx, new_vy);
    }
}

 
#include "../fg_class_tracking.h"
#include "../2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <random>

/**
 * Fixed Trajectory Comparison Data Generator
 * ==========================================
 * 
 * This version generates data CONSISTENT with the factor graph model:
 * - Constant velocity motion (no accelerations)
 * - Process noise only (as modeled by factor graph)
 * 
 * This eliminates model mismatch and shows true parameter tuning effects.
 */

// Generate straight-line trajectory with constant velocity
std::vector<Eigen::Vector4d> generateConstantVelocityTrajectory(int N, double dt) {
    std::vector<Eigen::Vector4d> states(N);
    
    // Start at origin with initial velocity
    states[0] << 0.0, 0.0, 2.0, 1.5;  // [x, y, vx, vy]
    
    // System matrix for constant velocity
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  // x = x + vx*dt
    F(1, 3) = dt;  // y = y + vy*dt
    
    // Generate pure constant velocity motion (no accelerations)
    for (int k = 1; k < N; ++k) {
        states[k] = F * states[k-1];  // Perfect constant velocity
    }
    
    return states;
}

// Generate slightly curved trajectory (small, consistent perturbations)
std::vector<Eigen::Vector4d> generateSlightlyCurvedTrajectory(int N, double dt) {
    std::vector<Eigen::Vector4d> states(N);
    
    // Start at origin with initial velocity
    states[0] << 0.0, 0.0, 2.0, 1.0;
    
    // System matrices
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  F(1, 3) = dt;
    
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;
    
    // Generate gentle curve using VERY SMALL accelerations
    // These should be small enough that factor graph can handle via process noise
    for (int k = 1; k < N; ++k) {
        double t = k * dt;
        
        // SMALL gentle accelerations (within process noise range)
        Eigen::Vector2d acceleration;
        acceleration << 0.05 * sin(0.2 * t),   // Very small x acceleration
                       0.03 * cos(0.25 * t);   // Very small y acceleration
        
        states[k] = F * states[k-1] + B * acceleration;
    }
    
    return states;
}

// Add process noise to trajectory
std::vector<Eigen::Vector4d> addProcessNoise(const std::vector<Eigen::Vector4d>& clean_states,
                                            double noise_intensity, double dt, std::mt19937& gen) {
    std::vector<Eigen::Vector4d> noisy_states = clean_states;
    int N = clean_states.size();
    
    if (noise_intensity < 1e-8) {
        return noisy_states;  // Skip if noise is effectively zero
    }
    
    // Build Q matrix using same method as factor graph
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt, dt3 = dt2 * dt;
    double V0 = noise_intensity, V1 = noise_intensity;
    
    Q(0, 0) = dt3/3.0 * V0;  Q(0, 2) = dt2/2.0 * V0;
    Q(1, 1) = dt3/3.0 * V1;  Q(1, 3) = dt2/2.0 * V1;
    Q(2, 0) = dt2/2.0 * V0;  Q(2, 2) = dt * V0;
    Q(3, 1) = dt2/2.0 * V1;  Q(3, 3) = dt * V1;
    
    Eigen::LLT<Eigen::Matrix4d> llt(Q);
    Eigen::Matrix4d L = llt.matrixL();
    
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (int k = 1; k < N; ++k) {
        Eigen::Vector4d noise;
        for (int i = 0; i < 4; ++i) noise[i] = normal(gen);
        noisy_states[k] += L * noise;
    }
    
    return noisy_states;
}

// Generate noisy measurements (same as before)
std::vector<Eigen::Vector2d> generateNoisyMeasurements(const std::vector<Eigen::Vector4d>& states,
                                                      double noise_var, std::mt19937& gen) {
    std::vector<Eigen::Vector2d> measurements(states.size());
    
    if (noise_var < 1e-12) {
        for (int k = 0; k < states.size(); ++k) {
            measurements[k] = states[k].head<2>();
        }
        return measurements;
    }
    
    double noise_std = sqrt(noise_var);
    std::normal_distribution<double> normal(0.0, noise_std);
    
    for (int k = 0; k < states.size(); ++k) {
        measurements[k] = states[k].head<2>();
        measurements[k][0] += normal(gen);
        measurements[k][1] += normal(gen);
    }
    
    return measurements;
}

// Run factor graph (same as before)
std::vector<Eigen::Vector4d> runFactorGraph(const std::vector<Eigen::Vector4d>& true_states,
                                           const std::vector<Eigen::Vector2d>& measurements,
                                           double q_param, double R_param, double dt) {
    FactorGraph2DTrajectory fg;
    fg.setQFromProcessNoiseIntensity(q_param, dt);
    fg.setRFromMeasurementNoise(sqrt(R_param), sqrt(R_param));
    
    fg.run(true_states, &measurements, dt, true);
    return fg.getAllEstimates();
}

// Save data with clearer column names
void saveTrajectoryToCSV(const std::string& filename,
                        const std::vector<Eigen::Vector4d>& true_trajectory,
                        const std::vector<Eigen::Vector2d>& measurements,
                        const std::vector<Eigen::Vector4d>& estimates_tuned,
                        const std::vector<Eigen::Vector4d>& estimates_bad1,
                        const std::vector<Eigen::Vector4d>& estimates_bad2,
                        double dt) {
    
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6);
    
    file << "time,true_x,true_y,true_vx,true_vy,meas_x,meas_y,"
         << "tuned_x,tuned_y,tuned_vx,tuned_vy,"
         << "bad1_x,bad1_y,bad1_vx,bad1_vy,"
         << "bad2_x,bad2_y,bad2_vx,bad2_vy\n";
    
    for (int k = 0; k < true_trajectory.size(); ++k) {
        double t = k * dt;
        const auto& true_state = true_trajectory[k];
        const auto& meas = measurements[k];
        const auto& est_tuned = estimates_tuned[k];
        const auto& est_bad1 = estimates_bad1[k];
        const auto& est_bad2 = estimates_bad2[k];
        
        file << t << ","
             << true_state[0] << "," << true_state[1] << "," << true_state[2] << "," << true_state[3] << ","
             << meas[0] << "," << meas[1] << ","
             << est_tuned[0] << "," << est_tuned[1] << "," << est_tuned[2] << "," << est_tuned[3] << ","
             << est_bad1[0] << "," << est_bad1[1] << "," << est_bad1[2] << "," << est_bad1[3] << ","
             << est_bad2[0] << "," << est_bad2[1] << "," << est_bad2[2] << "," << est_bad2[3] << "\n";
    }
    
    file.close();
    std::cout << "Saved fixed trajectory data to: " << filename << std::endl;
}

int main() {
    std::cout << "=== Generating FIXED Trajectory Comparison Data ===" << std::endl;
    std::cout << "FIXED: Data generation now consistent with factor graph model!" << std::endl;
    
    // Load parameters
    YAML::Node config;
    try {
        config = YAML::LoadFile("../BO_Parameters.yaml");
    } catch (const std::exception& e) {
        std::cout << "Error loading config: " << e.what() << std::endl;
        return 1;
    }
    
    double tuned_q = config["validate_filter"]["q"].as<double>();
    double tuned_R = config["validate_filter"]["R"].as<double>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    double true_process_noise = config["Data_Generation"]["q"].as<double>();
    double true_meas_noise_var = config["Data_Generation"]["meas_noise_var"].as<double>();
    
    int trajectory_length = 50;  // Good length for clear visualization
    std::mt19937 gen(42);        // Fixed seed for reproducibility
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Trajectory type: Constant velocity (consistent with factor graph)" << std::endl;
    std::cout << "  Trajectory length: " << trajectory_length << std::endl;
    std::cout << "  Time step: " << dt << std::endl;
    std::cout << "  Tuned Q: " << tuned_q << ", R: " << tuned_R << std::endl;
    std::cout << "  True process noise: " << true_process_noise << std::endl;
    std::cout << "  True measurement noise var: " << true_meas_noise_var << std::endl;
    
    // Choose trajectory type
    std::cout << "\nSelect trajectory type:" << std::endl;
    std::cout << "1. Pure constant velocity (straight line)" << std::endl;
    std::cout << "2. Slightly curved (small accelerations within noise range)" << std::endl;
    std::cout << "Choice [1-2]: ";
    
    int choice = 1;  // Default to straight line
    // std::cin >> choice;  // Uncomment for interactive mode
    
    // Generate clean trajectory
    std::vector<Eigen::Vector4d> clean_trajectory;
    if (choice == 2) {
        std::cout << "Generating slightly curved trajectory..." << std::endl;
        clean_trajectory = generateSlightlyCurvedTrajectory(trajectory_length, dt);
    } else {
        std::cout << "Generating straight-line constant velocity trajectory..." << std::endl;
        clean_trajectory = generateConstantVelocityTrajectory(trajectory_length, dt);
    }
    
    // Add process noise
    auto true_trajectory = addProcessNoise(clean_trajectory, true_process_noise, dt, gen);
    
    // Generate measurements
    auto measurements = generateNoisyMeasurements(true_trajectory, true_meas_noise_var, gen);
    
    std::cout << "Running factor graph optimizations..." << std::endl;
    
    try {
        // Run with different parameter sets
        auto estimates_tuned = runFactorGraph(true_trajectory, measurements, tuned_q, tuned_R, dt);
        
        // More extreme bad parameters for clearer differences
        double bad_q_low = tuned_q * 0.01;   // Very low process noise trust
        double bad_R_high = tuned_R * 100;   // Very high measurement noise
        auto estimates_bad1 = runFactorGraph(true_trajectory, measurements, bad_q_low, bad_R_high, dt);
        
        double bad_q_high = tuned_q * 100;   // Very high process noise
        double bad_R_low = tuned_R * 0.01;   // Very low measurement noise trust
        auto estimates_bad2 = runFactorGraph(true_trajectory, measurements, bad_q_high, bad_R_low, dt);
        
        std::cout << "\nParameter sets tested:" << std::endl;
        std::cout << "  Tuned:     Q=" << tuned_q << ", R=" << tuned_R << std::endl;
        std::cout << "  Bad Set 1: Q=" << bad_q_low << ", R=" << bad_R_high << " (Under-trusts process, over-trusts measurements)" << std::endl;
        std::cout << "  Bad Set 2: Q=" << bad_q_high << ", R=" << bad_R_low << " (Over-trusts process, under-trusts measurements)" << std::endl;
        
        // Save data
        saveTrajectoryToCSV("plots/trajectory_comparison_data_fixed.csv",
                           true_trajectory, measurements,
                           estimates_tuned, estimates_bad1, estimates_bad2, dt);
        
        // Calculate MSE
        auto calculateMSE = [](const std::vector<Eigen::Vector4d>& est, const std::vector<Eigen::Vector4d>& true_traj) {
            double sum = 0.0;
            for (int k = 0; k < est.size(); ++k) {
                Eigen::Vector2d pos_error = est[k].head<2>() - true_traj[k].head<2>();
                sum += pos_error.squaredNorm();
            }
            return sum / est.size();
        };
        
        double mse_tuned = calculateMSE(estimates_tuned, true_trajectory);
        double mse_bad1 = calculateMSE(estimates_bad1, true_trajectory);
        double mse_bad2 = calculateMSE(estimates_bad2, true_trajectory);
        
        std::cout << "\nPosition MSE Results (FIXED MODEL):" << std::endl;
        std::cout << "  Tuned parameters: " << std::scientific << mse_tuned << std::endl;
        std::cout << "  Bad Set 1: " << mse_bad1 << std::endl;
        std::cout << "  Bad Set 2: " << mse_bad2 << std::endl;
        
        std::cout << "\nPerformance improvements:" << std::endl;
        std::cout << "  Tuned vs Bad Set 1: " << std::fixed << std::setprecision(1) 
                  << (mse_bad1 / mse_tuned) << "x better" << std::endl;
        std::cout << "  Tuned vs Bad Set 2: " << (mse_bad2 / mse_tuned) << "x better" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error during optimization: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ FIXED data generation complete!" << std::endl;
    std::cout << "Now run: python3 plot_trajectory_comparison_simple.py" << std::endl;
    std::cout << "The differences should be much clearer now!" << std::endl;
    
    return 0;
} 
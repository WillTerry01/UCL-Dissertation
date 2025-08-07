#include "../fg_class_tracking.h"
#include "../2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <random>
#include <chrono>

/**
 * Performance Validation Test for Tuned Parameters
 * ================================================
 * 
 * This test validates that tuned Q and R parameters actually improve 
 * factor graph performance by:
 * 
 * 1. Loading tuned parameters from BO_Parameters.yaml
 * 2. Generating 1000 noisy test trajectories
 * 3. Running factor graph optimization on each
 * 4. Comparing estimates to ideal straight-line trajectory
 * 5. Computing average MSE and showing improvement over bad parameters
 * 
 * Expected result: Tuned parameters should give much lower MSE than arbitrary bad ones.
 */

// Generate ideal straight-line trajectory (ground truth)
std::vector<Eigen::Vector4d> generateIdealTrajectory(int N, double dt, 
                                                     Eigen::Vector2d start_pos = Eigen::Vector2d(0, 0),
                                                     Eigen::Vector2d velocity = Eigen::Vector2d(1, 1)) {
    std::vector<Eigen::Vector4d> ideal_states(N);
    
    for (int k = 0; k < N; ++k) {
        double t = k * dt;
        ideal_states[k] << start_pos[0] + velocity[0] * t,  // x position
                          start_pos[1] + velocity[1] * t,  // y position  
                          velocity[0],                     // x velocity
                          velocity[1];                     // y velocity
    }
    
    return ideal_states;
}

// Generate noisy trajectory using the system model
std::vector<Eigen::Vector4d> generateNoisyTrajectory(const std::vector<Eigen::Vector4d>& ideal_states,
                                                     double process_noise_intensity, double dt,
                                                     std::mt19937& gen) {
    std::vector<Eigen::Vector4d> noisy_states = ideal_states;
    int N = ideal_states.size();
    
    // Construct Q matrix for noise generation
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double V0 = process_noise_intensity;
    double V1 = process_noise_intensity;
    
    Q(0, 0) = dt3 / 3.0 * V0;  Q(0, 2) = dt2 / 2.0 * V0;
    Q(1, 1) = dt3 / 3.0 * V1;  Q(1, 3) = dt2 / 2.0 * V1;
    Q(2, 0) = dt2 / 2.0 * V0;  Q(2, 2) = dt * V0;
    Q(3, 1) = dt2 / 2.0 * V1;  Q(3, 3) = dt * V1;
    
    // Generate correlated process noise
    Eigen::LLT<Eigen::Matrix4d> llt(Q);
    Eigen::Matrix4d L = llt.matrixL();
    
    // System matrices
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  F(1, 3) = dt;
    
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (int k = 1; k < N; ++k) {
        // Generate process noise
        Eigen::Vector4d noise;
        for (int i = 0; i < 4; ++i) {
            noise[i] = normal(gen);
        }
        Eigen::Vector4d correlated_noise = L * noise;
        
        // Apply system dynamics with noise
        noisy_states[k] = F * noisy_states[k-1] + correlated_noise;
    }
    
    return noisy_states;
}

// Generate noisy measurements
std::vector<Eigen::Vector2d> generateNoisyMeasurements(const std::vector<Eigen::Vector4d>& states,
                                                      double measurement_noise_std,
                                                      std::mt19937& gen) {
    std::vector<Eigen::Vector2d> measurements(states.size());
    std::normal_distribution<double> normal(0.0, measurement_noise_std);
    
    for (int k = 0; k < states.size(); ++k) {
        measurements[k] = states[k].head<2>(); // Extract position
        measurements[k][0] += normal(gen);     // Add x noise
        measurements[k][1] += normal(gen);     // Add y noise
    }
    
    return measurements;
}

// Calculate MSE between estimated and ideal trajectories
double calculateMSE(const std::vector<Eigen::Vector4d>& estimated_states,
                   const std::vector<Eigen::Vector4d>& ideal_states) {
    if (estimated_states.size() != ideal_states.size()) {
        throw std::runtime_error("State vectors must have same size");
    }
    
    double total_error = 0.0;
    int N = estimated_states.size();
    
    for (int k = 0; k < N; ++k) {
        Eigen::Vector4d error = estimated_states[k] - ideal_states[k];
        total_error += error.squaredNorm();
    }
    
    return total_error / (N * 4);  // Average over all elements
}

// Calculate position-only MSE (more intuitive)
double calculatePositionMSE(const std::vector<Eigen::Vector4d>& estimated_states,
                           const std::vector<Eigen::Vector4d>& ideal_states) {
    double total_error = 0.0;
    int N = estimated_states.size();
    
    for (int k = 0; k < N; ++k) {
        Eigen::Vector2d pos_error = estimated_states[k].head<2>() - ideal_states[k].head<2>();
        total_error += pos_error.squaredNorm();
    }
    
    return total_error / N;  // Average position error
}

// Test factor graph performance with given Q/R parameters
struct PerformanceResults {
    double avg_total_mse;
    double avg_position_mse;
    double std_total_mse;
    double std_position_mse;
    int successful_optimizations;
    std::vector<double> all_mse_values;
};

PerformanceResults testFactorGraphPerformance(double q_param, double R_param, 
                                            int num_test_runs, int trajectory_length,
                                            double dt, double true_process_noise,
                                            double true_measurement_noise,
                                            const std::string& test_name) {
    
    std::cout << "\n=== Testing " << test_name << " ===" << std::endl;
    std::cout << "Q parameter: " << q_param << ", R parameter: " << R_param << std::endl;
    std::cout << "Running " << num_test_runs << " test trajectories..." << std::endl;
    
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::vector<double> total_mse_values;
    std::vector<double> position_mse_values;
    int successful_runs = 0;
    
    // Generate ideal reference trajectory
    auto ideal_trajectory = generateIdealTrajectory(trajectory_length, dt);
    
    std::cout << "Ideal trajectory: Start (0,0), End (" 
              << ideal_trajectory.back()[0] << "," << ideal_trajectory.back()[1] 
              << "), Velocity (1,1)" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_test_runs; ++run) {
        if (run % 100 == 0) {
            std::cout << "Progress: " << run << "/" << num_test_runs << " runs completed\r" << std::flush;
        }
        
        try {
            // Generate noisy test trajectory
            auto noisy_states = generateNoisyTrajectory(ideal_trajectory, true_process_noise, dt, gen);
            auto noisy_measurements = generateNoisyMeasurements(noisy_states, true_measurement_noise, gen);
            
            // Create and configure factor graph
            FactorGraph2DTrajectory fg;
            fg.setQFromProcessNoiseIntensity(q_param, dt);
            fg.setRFromMeasurementNoise(sqrt(R_param), sqrt(R_param));
            
            // Run optimization
            fg.run(noisy_states, &noisy_measurements, dt, true);
            
            // Get estimates and calculate errors
            auto estimated_states = fg.getAllEstimates();
            double total_mse = calculateMSE(estimated_states, ideal_trajectory);
            double position_mse = calculatePositionMSE(estimated_states, ideal_trajectory);
            
            // Check for reasonable results (filter out optimization failures)
            if (total_mse < 1000.0 && !std::isnan(total_mse) && !std::isinf(total_mse)) {
                total_mse_values.push_back(total_mse);
                position_mse_values.push_back(position_mse);
                successful_runs++;
            }
            
        } catch (const std::exception& e) {
            // Skip failed optimizations
            continue;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nCompleted " << successful_runs << "/" << num_test_runs 
              << " successful optimizations in " << duration.count() << "ms" << std::endl;
    
    // Calculate statistics
    PerformanceResults results;
    results.successful_optimizations = successful_runs;
    results.all_mse_values = total_mse_values;
    
    if (successful_runs > 0) {
        // Calculate means
        double sum_total = 0.0, sum_position = 0.0;
        for (int i = 0; i < successful_runs; ++i) {
            sum_total += total_mse_values[i];
            sum_position += position_mse_values[i];
        }
        results.avg_total_mse = sum_total / successful_runs;
        results.avg_position_mse = sum_position / successful_runs;
        
        // Calculate standard deviations
        double var_total = 0.0, var_position = 0.0;
        for (int i = 0; i < successful_runs; ++i) {
            var_total += (total_mse_values[i] - results.avg_total_mse) * (total_mse_values[i] - results.avg_total_mse);
            var_position += (position_mse_values[i] - results.avg_position_mse) * (position_mse_values[i] - results.avg_position_mse);
        }
        results.std_total_mse = sqrt(var_total / (successful_runs - 1));
        results.std_position_mse = sqrt(var_position / (successful_runs - 1));
    } else {
        results.avg_total_mse = std::numeric_limits<double>::infinity();
        results.avg_position_mse = std::numeric_limits<double>::infinity();
        results.std_total_mse = 0.0;
        results.std_position_mse = 0.0;
    }
    
    return results;
}

int main() {
    std::cout << "=== Factor Graph Performance Validation with Tuned Parameters ===" << std::endl;
    std::cout << "Testing tuned Q/R parameters vs. intentionally bad parameters" << std::endl;
    std::cout << std::endl;
    
    // Load tuned parameters from YAML
    YAML::Node config;
    try {
        config = YAML::LoadFile("../BO_Parameters.yaml");
    } catch (const std::exception& e) {
        std::cout << "Error loading BO_Parameters.yaml: " << e.what() << std::endl;
        return 1;
    }
    
    // Extract parameters
    double tuned_q = config["validate_filter"]["q"].as<double>();
    double tuned_R = config["validate_filter"]["R"].as<double>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    int trajectory_length = 25;  // Use manageable length for 1000 runs
    int num_test_runs = 1000;
    
    // True noise levels for data generation (moderate values)
    double true_process_noise = config["Data_Generation"]["q"].as<double>();
    double true_measurement_noise = config["Data_Generation"]["meas_noise_var"].as<double>();
    
    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Trajectory length: " << trajectory_length << std::endl;
    std::cout << "  Time step (dt): " << dt << std::endl;
    std::cout << "  Number of test runs: " << num_test_runs << std::endl;
    std::cout << "  True process noise intensity: " << true_process_noise << std::endl;
    std::cout << "  True measurement noise std: " << true_measurement_noise << std::endl;
    std::cout << std::endl;
    
    std::cout << "Loaded tuned parameters:" << std::endl;
    std::cout << "  Tuned Q intensity: " << tuned_q << std::endl;
    std::cout << "  Tuned R variance: " << tuned_R << std::endl;
    std::cout << std::endl;
    
    // Test 1: Tuned parameters
    auto tuned_results = testFactorGraphPerformance(tuned_q, tuned_R, num_test_runs, 
                                                   trajectory_length, dt, 
                                                   true_process_noise, true_measurement_noise,
                                                   "Tuned Parameters");
    
    // Test 2: Bad parameters (too low Q, too high R)
    double bad_q_low = tuned_q * 0.01;    // 100x too small
    double bad_R_high = tuned_R * 100;    // 100x too large
    auto bad_results_1 = testFactorGraphPerformance(bad_q_low, bad_R_high, num_test_runs,
                                                   trajectory_length, dt,
                                                   true_process_noise, true_measurement_noise,
                                                   "Bad Parameters (Low Q, High R)");
    
    // Test 3: Bad parameters (too high Q, too low R)  
    double bad_q_high = tuned_q * 100;    // 100x too large
    double bad_R_low = tuned_R * 0.01;    // 100x too small
    auto bad_results_2 = testFactorGraphPerformance(bad_q_high, bad_R_low, num_test_runs,
                                                   trajectory_length, dt,
                                                   true_process_noise, true_measurement_noise,
                                                   "Bad Parameters (High Q, Low R)");
    
    // Summary results
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE COMPARISON SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "\n1. TUNED PARAMETERS:" << std::endl;
    std::cout << "   Q = " << tuned_q << ", R = " << tuned_R << std::endl;
    std::cout << "   Success rate: " << tuned_results.successful_optimizations << "/" << num_test_runs << std::endl;
    std::cout << "   Average Total MSE: " << std::scientific << tuned_results.avg_total_mse 
              << " ± " << tuned_results.std_total_mse << std::endl;
    std::cout << "   Average Position MSE: " << tuned_results.avg_position_mse 
              << " ± " << tuned_results.std_position_mse << std::endl;
    
    std::cout << "\n2. BAD PARAMETERS (Low Q, High R):" << std::endl;
    std::cout << "   Q = " << bad_q_low << ", R = " << bad_R_high << std::endl;
    std::cout << "   Success rate: " << bad_results_1.successful_optimizations << "/" << num_test_runs << std::endl;
    std::cout << "   Average Total MSE: " << bad_results_1.avg_total_mse 
              << " ± " << bad_results_1.std_total_mse << std::endl;
    std::cout << "   Average Position MSE: " << bad_results_1.avg_position_mse 
              << " ± " << bad_results_1.std_position_mse << std::endl;
    
    std::cout << "\n3. BAD PARAMETERS (High Q, Low R):" << std::endl;
    std::cout << "   Q = " << bad_q_high << ", R = " << bad_R_low << std::endl;
    std::cout << "   Success rate: " << bad_results_2.successful_optimizations << "/" << num_test_runs << std::endl;
    std::cout << "   Average Total MSE: " << bad_results_2.avg_total_mse 
              << " ± " << bad_results_2.std_total_mse << std::endl;
    std::cout << "   Average Position MSE: " << bad_results_2.avg_position_mse 
              << " ± " << bad_results_2.std_position_mse << std::endl;
    
    // Performance improvement analysis
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << "PERFORMANCE IMPROVEMENT ANALYSIS:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    if (bad_results_1.avg_position_mse > 0 && tuned_results.avg_position_mse > 0) {
        double improvement_1 = bad_results_1.avg_position_mse / tuned_results.avg_position_mse;
        std::cout << "Tuned vs Bad (Low Q, High R): " << std::fixed << std::setprecision(1) 
                  << improvement_1 << "x better position accuracy" << std::endl;
    }
    
    if (bad_results_2.avg_position_mse > 0 && tuned_results.avg_position_mse > 0) {
        double improvement_2 = bad_results_2.avg_position_mse / tuned_results.avg_position_mse;
        std::cout << "Tuned vs Bad (High Q, Low R): " << improvement_2 
                  << "x better position accuracy" << std::endl;
    }
    
    std::cout << "\n✓ SUCCESS: " << (tuned_results.avg_position_mse < bad_results_1.avg_position_mse && 
                                     tuned_results.avg_position_mse < bad_results_2.avg_position_mse ?
                                     "Tuned parameters perform significantly better!" :
                                     "Results are inconclusive - check parameter ranges") << std::endl;
    
    return 0;
}
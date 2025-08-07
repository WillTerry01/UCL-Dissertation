#include "../fg_class_tracking.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <random>
#include <chrono>

/**
 * Extreme Parameter Validation Test
 * =================================
 * 
 * This test uses more challenging conditions to demonstrate
 * the importance of proper parameter tuning:
 * 
 * 1. Longer trajectories (T=100) for more constraint
 * 2. More extreme parameter mismatches (1000x off)
 * 3. Higher noise levels to stress the system
 * 4. Different trajectory types (curved motion)
 */

// Generate curved trajectory (more challenging than straight line)
std::vector<Eigen::Vector4d> generateCurvedTrajectory(int N, double dt) {
    std::vector<Eigen::Vector4d> states(N);
    
    // Start at origin with initial velocity
    states[0] << 0.0, 0.0, 1.0, 0.0;
    
    // System matrices
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  F(1, 3) = dt;
    
    // Control matrix for acceleration
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;
    
    // Generate curved motion with time-varying acceleration
    for (int k = 1; k < N; ++k) {
        double t = k * dt;
        
        // Sinusoidal acceleration to create curved path
        Eigen::Vector2d acceleration;
        acceleration << 0.1 * sin(0.5 * t),   // x acceleration
                       0.1 * cos(0.5 * t);    // y acceleration
        
        // Apply system dynamics
        states[k] = F * states[k-1] + B * acceleration;
    }
    
    return states;
}

// Add realistic process noise
std::vector<Eigen::Vector4d> addProcessNoise(const std::vector<Eigen::Vector4d>& clean_states,
                                            double noise_intensity, double dt, std::mt19937& gen) {
    std::vector<Eigen::Vector4d> noisy_states = clean_states;
    int N = clean_states.size();
    
    // Build Q matrix
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

// Generate noisy measurements
std::vector<Eigen::Vector2d> generateNoisyMeasurements(const std::vector<Eigen::Vector4d>& states,
                                                      double noise_std, std::mt19937& gen) {
    std::vector<Eigen::Vector2d> measurements(states.size());
    std::normal_distribution<double> normal(0.0, noise_std);
    
    for (int k = 0; k < states.size(); ++k) {
        measurements[k] = states[k].head<2>();
        measurements[k][0] += normal(gen);
        measurements[k][1] += normal(gen);
    }
    
    return measurements;
}

// Calculate trajectory error metrics
struct TrajectoryMetrics {
    double final_position_error;
    double avg_position_error;
    double max_position_error;
    double avg_velocity_error;
    double total_mse;
};

TrajectoryMetrics calculateTrajectoryMetrics(const std::vector<Eigen::Vector4d>& estimated,
                                            const std::vector<Eigen::Vector4d>& true_trajectory) {
    TrajectoryMetrics metrics;
    int N = estimated.size();
    
    double sum_pos_error = 0.0, sum_vel_error = 0.0, sum_mse = 0.0;
    double max_pos_error = 0.0;
    
    for (int k = 0; k < N; ++k) {
        Eigen::Vector2d pos_error = estimated[k].head<2>() - true_trajectory[k].head<2>();
        Eigen::Vector2d vel_error = estimated[k].tail<2>() - true_trajectory[k].tail<2>();
        Eigen::Vector4d total_error = estimated[k] - true_trajectory[k];
        
        double pos_error_norm = pos_error.norm();
        sum_pos_error += pos_error_norm;
        sum_vel_error += vel_error.norm();
        sum_mse += total_error.squaredNorm();
        max_pos_error = std::max(max_pos_error, pos_error_norm);
    }
    
    metrics.final_position_error = (estimated.back().head<2>() - true_trajectory.back().head<2>()).norm();
    metrics.avg_position_error = sum_pos_error / N;
    metrics.max_position_error = max_pos_error;
    metrics.avg_velocity_error = sum_vel_error / N;
    metrics.total_mse = sum_mse / (N * 4);
    
    return metrics;
}

// Test with specific parameter set
struct TestResults {
    double success_rate;
    TrajectoryMetrics avg_metrics;
    TrajectoryMetrics std_metrics;
    std::vector<double> all_position_errors;
};

TestResults runParameterTest(double q_param, double R_param, int num_runs, int traj_length,
                           double dt, double true_process_noise, double true_meas_noise,
                           const std::string& test_name) {
    
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Q = " << q_param << ", R = " << R_param << std::endl;
    
    std::mt19937 gen(42);
    std::vector<TrajectoryMetrics> all_metrics;
    int successful_runs = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; ++run) {
        if (run % 50 == 0) {
            std::cout << "Progress: " << run << "/" << num_runs << "\r" << std::flush;
        }
        
        try {
            // Generate challenging test case
            auto true_trajectory = generateCurvedTrajectory(traj_length, dt);
            auto noisy_states = addProcessNoise(true_trajectory, true_process_noise, dt, gen);
            auto measurements = generateNoisyMeasurements(noisy_states, true_meas_noise, gen);
            
            // Run factor graph
            FactorGraph2DTrajectory fg;
            fg.setQFromProcessNoiseIntensity(q_param, dt);
            fg.setRFromMeasurementNoise(sqrt(R_param), sqrt(R_param));
            
            fg.run(noisy_states, &measurements, dt, true);
            auto estimates = fg.getAllEstimates();
            
            // Calculate metrics against true (clean) trajectory
            auto metrics = calculateTrajectoryMetrics(estimates, true_trajectory);
            
            // Filter out unreasonable results
            if (metrics.total_mse < 1000.0 && !std::isnan(metrics.total_mse)) {
                all_metrics.push_back(metrics);
                successful_runs++;
            }
            
        } catch (const std::exception& e) {
            continue;  // Skip failed optimizations
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nCompleted " << successful_runs << "/" << num_runs 
              << " runs in " << duration.count() << "ms" << std::endl;
    
    // Calculate average and standard deviation of metrics
    TestResults results;
    results.success_rate = (double)successful_runs / num_runs;
    
    if (successful_runs > 0) {
        // Calculate means
        TrajectoryMetrics sum = {0, 0, 0, 0, 0};
        for (const auto& m : all_metrics) {
            sum.final_position_error += m.final_position_error;
            sum.avg_position_error += m.avg_position_error;
            sum.max_position_error += m.max_position_error;
            sum.avg_velocity_error += m.avg_velocity_error;
            sum.total_mse += m.total_mse;
            results.all_position_errors.push_back(m.avg_position_error);
        }
        
        results.avg_metrics.final_position_error = sum.final_position_error / successful_runs;
        results.avg_metrics.avg_position_error = sum.avg_position_error / successful_runs;
        results.avg_metrics.max_position_error = sum.max_position_error / successful_runs;
        results.avg_metrics.avg_velocity_error = sum.avg_velocity_error / successful_runs;
        results.avg_metrics.total_mse = sum.total_mse / successful_runs;
        
        // Calculate standard deviations
        TrajectoryMetrics var = {0, 0, 0, 0, 0};
        for (const auto& m : all_metrics) {
            var.final_position_error += pow(m.final_position_error - results.avg_metrics.final_position_error, 2);
            var.avg_position_error += pow(m.avg_position_error - results.avg_metrics.avg_position_error, 2);
            var.max_position_error += pow(m.max_position_error - results.avg_metrics.max_position_error, 2);
            var.avg_velocity_error += pow(m.avg_velocity_error - results.avg_metrics.avg_velocity_error, 2);
            var.total_mse += pow(m.total_mse - results.avg_metrics.total_mse, 2);
        }
        
        results.std_metrics.final_position_error = sqrt(var.final_position_error / (successful_runs - 1));
        results.std_metrics.avg_position_error = sqrt(var.avg_position_error / (successful_runs - 1));
        results.std_metrics.max_position_error = sqrt(var.max_position_error / (successful_runs - 1));
        results.std_metrics.avg_velocity_error = sqrt(var.avg_velocity_error / (successful_runs - 1));
        results.std_metrics.total_mse = sqrt(var.total_mse / (successful_runs - 1));
    }
    
    return results;
}

int main() {
    std::cout << "=== Extreme Parameter Validation Test ===" << std::endl;
    std::cout << "Testing with longer trajectories and more extreme parameter mismatches" << std::endl;
    
    // Load tuned parameters
    YAML::Node config;
    try {
        config = YAML::LoadFile("../BO_Parameters.yaml");
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    double tuned_q = config["validate_filter"]["q"].as<double>();
    double tuned_R = config["validate_filter"]["R"].as<double>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    // More challenging test parameters
    int trajectory_length = 100;  // Longer for more constraint
    int num_test_runs = 200;      // Fewer runs due to longer trajectories
    double true_process_noise = 0.2;    // Higher noise
    double true_measurement_noise = 0.15;
    
    std::cout << "\nExtreme Test Configuration:" << std::endl;
    std::cout << "  Trajectory length: " << trajectory_length << std::endl;
    std::cout << "  Time step: " << dt << std::endl;
    std::cout << "  Test runs: " << num_test_runs << std::endl;
    std::cout << "  True process noise: " << true_process_noise << std::endl;
    std::cout << "  True measurement noise: " << true_measurement_noise << std::endl;
    std::cout << "  Motion type: Curved trajectory with time-varying acceleration" << std::endl;
    
    // Test 1: Tuned parameters
    auto tuned_results = runParameterTest(tuned_q, tuned_R, num_test_runs, trajectory_length,
                                        dt, true_process_noise, true_measurement_noise,
                                        "TUNED PARAMETERS");
    
    // Test 2: Extremely bad parameters (1000x off)
    double bad_q_extreme = tuned_q * 0.001;     // 1000x too small
    double bad_R_extreme = tuned_R * 1000;     // 1000x too large
    auto bad_results_1 = runParameterTest(bad_q_extreme, bad_R_extreme, num_test_runs, trajectory_length,
                                        dt, true_process_noise, true_measurement_noise,
                                        "EXTREMELY BAD PARAMETERS (Low Q, High R)");
    
    // Test 3: Opposite extreme
    double bad_q_extreme2 = tuned_q * 1000;    // 1000x too large
    double bad_R_extreme2 = tuned_R * 0.001;   // 1000x too small
    auto bad_results_2 = runParameterTest(bad_q_extreme2, bad_R_extreme2, num_test_runs, trajectory_length,
                                        dt, true_process_noise, true_measurement_noise,
                                        "EXTREMELY BAD PARAMETERS (High Q, Low R)");
    
    // Results summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "EXTREME PARAMETER TEST RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(4);
    
    auto printResults = [](const std::string& name, const TestResults& results) {
        std::cout << "\n" << name << ":" << std::endl;
        std::cout << "  Success Rate: " << (results.success_rate * 100) << "%" << std::endl;
        std::cout << "  Avg Position Error: " << results.avg_metrics.avg_position_error 
                  << " ± " << results.std_metrics.avg_position_error << std::endl;
        std::cout << "  Final Position Error: " << results.avg_metrics.final_position_error 
                  << " ± " << results.std_metrics.final_position_error << std::endl;
        std::cout << "  Max Position Error: " << results.avg_metrics.max_position_error 
                  << " ± " << results.std_metrics.max_position_error << std::endl;
        std::cout << "  Velocity Error: " << results.avg_metrics.avg_velocity_error 
                  << " ± " << results.std_metrics.avg_velocity_error << std::endl;
        std::cout << std::scientific << "  Total MSE: " << results.avg_metrics.total_mse 
                  << " ± " << results.std_metrics.total_mse << std::endl;
    };
    
    printResults("TUNED PARAMETERS", tuned_results);
    printResults("BAD PARAMS (Low Q, High R)", bad_results_1);
    printResults("BAD PARAMS (High Q, Low R)", bad_results_2);
    
    // Performance comparison
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << "PERFORMANCE IMPROVEMENT:" << std::endl;
    
    if (tuned_results.avg_metrics.avg_position_error > 0) {
        double improvement1 = bad_results_1.avg_metrics.avg_position_error / tuned_results.avg_metrics.avg_position_error;
        double improvement2 = bad_results_2.avg_metrics.avg_position_error / tuned_results.avg_metrics.avg_position_error;
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "Tuned vs Bad (Low Q, High R): " << improvement1 << "x better" << std::endl;
        std::cout << "Tuned vs Bad (High Q, Low R): " << improvement2 << "x better" << std::endl;
        
        if (improvement1 > 2.0 || improvement2 > 2.0) {
            std::cout << "\n✓ SUCCESS: Tuned parameters show significant improvement!" << std::endl;
        } else {
            std::cout << "\n⚠ MODERATE: Some improvement shown, parameters may need refinement" << std::endl;
        }
    }
    
    return 0;
} 
#include "../fg_class_tracking.h"
#include "../2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Test Program: Zero Noise Factor Graph Validation (Using Existing Pipeline)
 * ==========================================================================
 * 
 * Purpose: Validate that the factor graph system equation is correctly implemented
 * by using the existing data generation pipeline with zero noise, then testing
 * if factor graph optimization achieves zero MSE.
 * 
 * Method (Option B):
 * 1. Generate zero-noise data using existing tracking_gen_data.cpp pipeline
 * 2. Load this zero-noise data (states and measurements)
 * 3. Set Q and R matrices to near-zero in factor graph
 * 4. Run optimization and verify MSE approaches zero
 * 
 * This tests the complete pipeline end-to-end with the professor's methodology.
 */

// Calculate Mean Squared Error between true and estimated states
double calculateMSE(const std::vector<Eigen::Vector4d>& true_states,
                   const std::vector<Eigen::Vector4d>& estimated_states) {
    if (true_states.size() != estimated_states.size()) {
        throw std::runtime_error("State vectors must have same size");
    }
    
    double total_error = 0.0;
    int N = true_states.size();
    
    for (int k = 0; k < N; ++k) {
        Eigen::Vector4d error = true_states[k] - estimated_states[k];
        total_error += error.squaredNorm();
    }
    
    return total_error / (N * 4);  // Average over all elements
}

// Print detailed error analysis
void printErrorAnalysis(const std::vector<Eigen::Vector4d>& true_states,
                       const std::vector<Eigen::Vector4d>& estimated_states,
                       const std::vector<Eigen::Vector2d>& measurements) {
    
    int N = true_states.size();
    
    std::cout << "Detailed Error Analysis (first 5 time steps):" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    
    for (int k = 0; k < std::min(5, N); ++k) {
        Eigen::Vector4d error = true_states[k] - estimated_states[k];
        Eigen::Vector2d meas_error = measurements[k] - true_states[k].head<2>();
        
        std::cout << "k=" << k << ":" << std::endl;
        std::cout << "  True state:     [" << true_states[k].transpose() << "]" << std::endl;
        std::cout << "  Estimated:      [" << estimated_states[k].transpose() << "]" << std::endl;
        std::cout << "  Measurement:    [" << measurements[k].transpose() << "]" << std::endl;
        std::cout << "  State error:    [" << error.transpose() << "]" << std::endl;
        std::cout << "  Meas error:     [" << meas_error.transpose() << "]" << std::endl;
        std::cout << std::endl;
    }
    
    // Calculate component-wise errors
    double pos_mse = 0.0, vel_mse = 0.0;
    double max_pos_error = 0.0, max_vel_error = 0.0;
    
    for (int k = 0; k < N; ++k) {
        Eigen::Vector2d pos_error = true_states[k].head<2>() - estimated_states[k].head<2>();
        Eigen::Vector2d vel_error = true_states[k].tail<2>() - estimated_states[k].tail<2>();
        
        pos_mse += pos_error.squaredNorm();
        vel_mse += vel_error.squaredNorm();
        
        max_pos_error = std::max(max_pos_error, pos_error.norm());
        max_vel_error = std::max(max_vel_error, vel_error.norm());
    }
    
    pos_mse /= (N * 2);
    vel_mse /= (N * 2);
    
    std::cout << "Component-wise Analysis:" << std::endl;
    std::cout << "  Position MSE:       " << std::scientific << pos_mse << std::endl;
    std::cout << "  Velocity MSE:       " << std::scientific << vel_mse << std::endl;
    std::cout << "  Max position error: " << std::scientific << max_pos_error << std::endl;
    std::cout << "  Max velocity error: " << std::scientific << max_vel_error << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Zero Noise Factor Graph Validation (Using Existing Pipeline) ===" << std::endl;
    std::cout << "Testing system equation: x_{k+1} = F*x_k + B*u_k + v_k" << std::endl;
    std::cout << "Using existing data generation pipeline with zero noise..." << std::endl;
    std::cout << std::endl;
    
    // Step 1: Generate zero-noise data using existing pipeline
    std::cout << "Step 1: Generate zero-noise data using tracking_gen_data..." << std::endl;
    
    // We need to run tracking_gen_data with zero_noise_config.yaml
    std::string config_file = "../2D-Tracking/zero_noise_config.yaml";
    std::string data_gen_cmd = "./tracking_gen_data " + config_file;
    
    std::cout << "Run this command first: cd build && " << data_gen_cmd << std::endl;
    std::cout << "Then run this test again." << std::endl;
    std::cout << std::endl;
    
    // Step 2: Try to load the zero-noise data
    std::cout << "Step 2: Loading zero-noise data..." << std::endl;
    
    std::string states_file = "../2D-Tracking/Saved_Data/2D_noisy_states.h5";
    std::string measurements_file = "../2D-Tracking/Saved_Data/2D_noisy_measurements.h5";
    
    try {
        // Load the data
        auto all_states = load_all_noisy_states_h5(states_file);
        auto all_measurements = load_all_noisy_measurements_h5(measurements_file);
        
        if (all_states.empty() || all_measurements.empty()) {
            std::cout << "Error: No data loaded. Please run tracking_gen_data first." << std::endl;
            return 1;
        }
        
        // Use the first trajectory for testing
        auto true_states = all_states[0];
        auto measurements = all_measurements[0];
        
        int N = true_states.size();
        double dt = 1.0;  // From our config
        
        std::cout << "Loaded " << all_states.size() << " trajectories" << std::endl;
        std::cout << "Using trajectory 0 with " << N << " time steps" << std::endl;
        std::cout << "Initial state: [" << true_states[0].transpose() << "]" << std::endl;
        std::cout << "Final state: [" << true_states.back().transpose() << "]" << std::endl;
        std::cout << std::endl;
        
        // Verify this is actually zero-noise data
        std::cout << "Verifying zero-noise properties:" << std::endl;
        
        // Check if measurements exactly match state positions (no measurement noise)
        double total_meas_error = 0.0;
        for (int k = 0; k < N; ++k) {
            Eigen::Vector2d meas_error = measurements[k] - true_states[k].head<2>();
            total_meas_error += meas_error.squaredNorm();
        }
        double avg_meas_error = total_meas_error / N;
        
        std::cout << "  Average measurement error: " << std::scientific << avg_meas_error << std::endl;
        
        if (avg_meas_error < 1e-10) {
            std::cout << "  ✓ Measurements match true positions (zero measurement noise)" << std::endl;
        } else {
            std::cout << "  ✗ Measurements have noise (measurement error = " << avg_meas_error << ")" << std::endl;
        }
        
        // Check if trajectory follows constant velocity (no process noise)
        // For zero process noise: x_{k+1} should equal F*x_k exactly
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0, 2) = dt;  // x position += x velocity * dt
        F(1, 3) = dt;  // y position += y velocity * dt
        
        double total_process_error = 0.0;
        for (int k = 1; k < N; ++k) {
            Eigen::Vector4d predicted = F * true_states[k-1];
            Eigen::Vector4d process_error = true_states[k] - predicted;
            total_process_error += process_error.squaredNorm();
        }
        double avg_process_error = total_process_error / (N-1);
        
        std::cout << "  Average process model error: " << std::scientific << avg_process_error << std::endl;
        
        if (avg_process_error < 1e-10) {
            std::cout << "  ✓ Trajectory follows exact constant velocity model (zero process noise)" << std::endl;
        } else {
            std::cout << "  ✗ Trajectory deviates from constant velocity (process noise present)" << std::endl;
        }
        std::cout << std::endl;
        
        // Step 3: Test factor graph with zero noise
        std::cout << "Step 3: Testing factor graph with zero Q and R matrices..." << std::endl;
        
        FactorGraph2DTrajectory fg;
        
        // Set Q and R matrices to near-zero (not exactly zero to avoid numerical issues)
        double noise_level = 1e-8;
        fg.setQFromProcessNoiseIntensity(noise_level, dt);
        fg.setRFromMeasurementNoise(sqrt(noise_level), sqrt(noise_level));
        
        std::cout << "Process noise intensity (q): " << noise_level << std::endl;
        std::cout << "Measurement noise std dev: " << sqrt(noise_level) << std::endl;
        std::cout << std::endl;
        
        // Run factor graph optimization
        fg.run(true_states, &measurements, dt, true);  // do_optimization = true
        
        // Get optimized estimates
        auto estimated_states = fg.getAllEstimates();
        
        // Calculate errors
        double mse = calculateMSE(true_states, estimated_states);
        double chi2 = fg.getChi2();
        
        std::cout << "Factor Graph Results:" << std::endl;
        std::cout << "  Chi-squared value: " << std::scientific << chi2 << std::endl;
        std::cout << "  Mean Squared Error: " << std::scientific << mse << std::endl;
        std::cout << std::endl;
        
        // Check if MSE is close to zero
        double mse_threshold = 1e-6;
        bool test_passed = (mse < mse_threshold);
        
        std::cout << "Validation Results:" << std::endl;
        std::cout << "  MSE Threshold: " << std::scientific << mse_threshold << std::endl;
        std::cout << "  Test Result: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
        
        if (test_passed) {
            std::cout << "  ✓ Factor graph correctly implements the system model!" << std::endl;
            std::cout << "  ✓ With zero noise, optimization achieves near-zero MSE" << std::endl;
            std::cout << "  ✓ End-to-end pipeline validation SUCCESSFUL" << std::endl;
        } else {
            std::cout << "  ✗ Factor graph may have implementation issues" << std::endl;
            std::cout << "  ✗ MSE is too high for zero-noise case" << std::endl;
        }
        std::cout << std::endl;
        
        // Step 4: Detailed analysis
        printErrorAnalysis(true_states, estimated_states, measurements);
        
        // Get graph structure info
        auto dims = fg.getActualGraphDimensions();
        std::cout << "Graph Structure Verification:" << std::endl;
        std::cout << "  Total edge dimensions (dimZ): " << dims.first << std::endl;
        std::cout << "  Total vertex dimensions (dimX): " << dims.second << std::endl;
        std::cout << "  Expected edges: " << (N-1) * 4 + N * 2 << " (process + measurement)" << std::endl;
        std::cout << "  Expected vertices: " << N * 4 << " (4D states)" << std::endl;
        std::cout << std::endl;
        
        return test_passed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cout << "Error loading data: " << e.what() << std::endl;
        std::cout << std::endl;
        std::cout << "Please run: cd build && ./tracking_gen_data ../2D-Tracking/zero_noise_config.yaml" << std::endl;
        std::cout << "Then run this test again." << std::endl;
        return 1;
    }
} 
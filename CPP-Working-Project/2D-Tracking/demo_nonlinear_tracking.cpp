/*
 * 2D Nonlinear Factor Graph Tracking Demo
 * =======================================
 * 
 * PURPOSE:
 * This demo tests the nonlinear factor graph system with:
 * 1. Constant Turn Rate (CT) motion model
 * 2. GPS Cartesian position (x, y) measurement model
 * 
 * This demonstrates the transition from linear to nonlinear factor graphs
 * and shows how the system handles curved trajectories with GPS measurements.
 */

#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "H5Cpp.h"
#include <vector>

int main() {
    std::cout << "=== 2D Nonlinear Factor Graph Tracking Demo ===" << std::endl;
    
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");
    
    // Nonlinear parameters
    double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);  // rad/s
    // Removed sensor_pos - not needed for GPS tracking
    
    std::cout << "=== Nonlinear Factor Graph Tracking Demo ===" << std::endl;
    std::cout << "  Turn rate: " << turn_rate << " rad/s" << std::endl;
    std::cout << "  Motion model: Constant Turn Rate" << std::endl;
    std::cout << "  Measurement model: GPS Cartesian position (x, y)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    std::cout << "Nonlinear System Parameters:" << std::endl;
    std::cout << "  Turn rate: " << turn_rate << " rad/s" << std::endl;
    std::cout << "  Time step (dt): " << dt << std::endl;
    std::cout << std::endl;
    
    // Load nonlinear data
    std::string states_file = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
    std::string measurements_file = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";
    
    std::cout << "Loading nonlinear data..." << std::endl;
    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    
    try {
        all_states = load_all_noisy_states_h5(states_file);
        all_measurements = load_all_noisy_measurements_h5(measurements_file);
        std::cout << "Successfully loaded " << all_states.size() << " nonlinear trajectories" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        std::cerr << "Please run tracking_gen_data_nonlinear.cpp first to generate nonlinear data." << std::endl;
        return -1;
    }
    
    // Test parameters
    int num_test_runs = std::min(3, static_cast<int>(all_states.size()));
    // Load optimized parameters from validation results
    double q_intensity = config["validate_filter"]["q"].as<double>();
    double meas_noise_var = config["validate_filter"]["R"].as<double>();
    double meas_noise_std = std::sqrt(meas_noise_var);
    
    // Get the consistency method that was used in BO optimization
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>();
    
    std::cout << "Testing nonlinear factor graph on " << num_test_runs << " trajectories..." << std::endl;
    std::cout << "Optimized process noise intensity (q): " << q_intensity << std::endl;
    std::cout << "Optimized measurement noise variance (R): " << meas_noise_var << std::endl;
    std::cout << "Measurement noise std: " << meas_noise_std << std::endl;
    std::cout << "Consistency method used in BO: " << consistency_method << std::endl;
    std::cout << std::endl;
    
    // Prepare results storage
    std::vector<std::vector<Eigen::Vector4d>> test_true_states;
    std::vector<std::vector<Eigen::Vector2d>> test_measurements;
    std::vector<std::vector<Eigen::Vector4d>> test_estimates;
    std::vector<double> test_chi2_values;
    std::vector<double> test_consistency_metrics;
    
    for (int run = 0; run < num_test_runs; ++run) {
        std::cout << "Processing nonlinear trajectory " << (run + 1) << "/" << num_test_runs << "..." << std::endl;
        
        // Create and configure nonlinear factor graph
        FactorGraph2DTrajectory fg;
        
        // Set nonlinear motion model
        fg.setMotionModelType("constant_turn_rate", turn_rate);
        
        // Set nonlinear measurement model
        fg.setMeasurementModelType("gps");  // GPS Cartesian position measurements
        
        // Set Q matrix using process noise intensity
        fg.setQFromProcessNoiseIntensity(q_intensity, dt);
        
        // Set R matrix for GPS measurements (x, y position)
        fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std); // x and y measurement noise
        
        // Configure output options
        FactorGraph2DTrajectory::OutputOptions opts;
        opts.output_estimated_state = true;
        opts.output_true_state = true;
        opts.output_information_matrix = true;
        fg.setOutputOptions(opts);
        
        if (consistency_method == "nis3") {
            // Run nonlinear optimization using NIS3 method (no optimization, perfect initialization)
            fg.runNonlinear(all_states[run], &all_measurements[run], dt, false);  // do_optimization = false for NIS3
        } else {
            // Run nonlinear optimization using NIS4 method (optimization, imperfect initialization)
            fg.runNonlinear(all_states[run], &all_measurements[run], dt, true);  // do_optimization = true for NIS4
        }
        
        // Get results
        auto est_states = fg.getAllEstimates();
        auto true_states = fg.getAllTrueStates();
        double chi2 = fg.getChi2();
        
        // Calculate consistency metrics (same as used in BO optimization)
        double consistency_metric = 0.0;
        int T = true_states.size();
        int nx = 4; // state dimension
        int nz = 2; // measurement dimension
        
        if (consistency_method == "cnees") {
            // Calculate CNEES (Normalized Estimation Error Squared)
            Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
            if (Hessian.rows() > 0 && Hessian.cols() > 0) {
                // Create full error vector
                Eigen::VectorXd full_error(T * nx);
                for (int k = 0; k < T; ++k) {
                    Eigen::Vector4d err = true_states[k] - est_states[k];
                    full_error.segment<4>(k * nx) = err;
                }
                // Calculate NEES: err^T * P^(-1) * err = err^T * H * err
                double nees = full_error.transpose() * Hessian * full_error;
                // Get actual graph dimensions for DOF calculation
                auto [dimZ, dimX] = fg.getActualGraphDimensions();
                int total_dof = dimX; // For NEES, DOF = total vertex dimensions
                consistency_metric = nees / total_dof;
            }
        } else if (consistency_method == "nis3" || consistency_method == "nis4") {
            // For NIS, chi2 already represents the measurement innovation
            // Get actual graph dimensions for DOF calculation
            auto [dimZ, dimX] = fg.getActualGraphDimensions();
            int total_dof = dimZ; // For NIS, DOF = total edge dimensions
            consistency_metric = chi2 / total_dof;
        }
        
        // Store results
        test_true_states.push_back(true_states);
        test_measurements.push_back(all_measurements[run]);
        test_estimates.push_back(est_states);
        test_chi2_values.push_back(chi2);
        test_consistency_metrics.push_back(consistency_metric);
        
        // Print statistics for this run
        double mean_position_error = 0.0;
        double mean_velocity_error = 0.0;
        
        for (int k = 0; k < T; ++k) {
            // Position error
            double pos_error = (est_states[k].head<2>() - true_states[k].head<2>()).norm();
            mean_position_error += pos_error;
            
            // Velocity error
            double vel_error = (est_states[k].tail<2>() - true_states[k].tail<2>()).norm();
            mean_velocity_error += vel_error;
        }
        
        mean_position_error /= T;
        mean_velocity_error /= T;
        
        std::cout << "  Run " << (run + 1) << " Results:" << std::endl;
        std::cout << "    Mean position error: " << mean_position_error << " m" << std::endl;
        std::cout << "    Mean velocity error: " << mean_velocity_error << " m/s" << std::endl;
        std::cout << "    Chi-squared: " << chi2 << std::endl;
        std::cout << "    Consistency Metric: " << consistency_metric << std::endl;
        std::cout << std::endl;
    }
    
    // Save results to HDF5
    std::string output_file = "../2D-Tracking/Saved_Data/nonlinear_tracking_results.h5";
    std::cout << "Saving results to " << output_file << std::endl;
    
    try {
        H5::H5File file(output_file, H5F_ACC_TRUNC);
        
        // Save true states
        int T = test_true_states[0].size();
        std::vector<double> true_states_data(num_test_runs * T * 4);
        for (int run = 0; run < num_test_runs; ++run) {
            for (int k = 0; k < T; ++k) {
                int idx = run * T * 4 + k * 4;
                true_states_data[idx + 0] = test_true_states[run][k][0];
                true_states_data[idx + 1] = test_true_states[run][k][1];
                true_states_data[idx + 2] = test_true_states[run][k][2];
                true_states_data[idx + 3] = test_true_states[run][k][3];
            }
        }
        hsize_t true_dims[3] = {static_cast<hsize_t>(num_test_runs), static_cast<hsize_t>(T), 4};
        H5::DataSpace true_space(3, true_dims);
        H5::DataSet true_dataset = file.createDataSet("true_states", H5::PredType::NATIVE_DOUBLE, true_space);
        true_dataset.write(true_states_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save measurements
        std::vector<double> measurements_data(num_test_runs * T * 2);
        for (int run = 0; run < num_test_runs; ++run) {
            for (int k = 0; k < T; ++k) {
                int idx = run * T * 2 + k * 2;
                measurements_data[idx + 0] = test_measurements[run][k][0];
                measurements_data[idx + 1] = test_measurements[run][k][1];
            }
        }
        hsize_t meas_dims[3] = {static_cast<hsize_t>(num_test_runs), static_cast<hsize_t>(T), 2};
        H5::DataSpace meas_space(3, meas_dims);
        H5::DataSet meas_dataset = file.createDataSet("measurements", H5::PredType::NATIVE_DOUBLE, meas_space);
        meas_dataset.write(measurements_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save estimates
        std::vector<double> estimates_data(num_test_runs * T * 4);
        for (int run = 0; run < num_test_runs; ++run) {
            for (int k = 0; k < T; ++k) {
                int idx = run * T * 4 + k * 4;
                estimates_data[idx + 0] = test_estimates[run][k][0];
                estimates_data[idx + 1] = test_estimates[run][k][1];
                estimates_data[idx + 2] = test_estimates[run][k][2];
                estimates_data[idx + 3] = test_estimates[run][k][3];
            }
        }
        H5::DataSet est_dataset = file.createDataSet("estimates", H5::PredType::NATIVE_DOUBLE, true_space);
        est_dataset.write(estimates_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save chi2 values
        hsize_t chi2_dims[1] = {static_cast<hsize_t>(num_test_runs)};
        H5::DataSpace chi2_space(1, chi2_dims);
        H5::DataSet chi2_dataset = file.createDataSet("chi2_values", H5::PredType::NATIVE_DOUBLE, chi2_space);
        chi2_dataset.write(test_chi2_values.data(), H5::PredType::NATIVE_DOUBLE);

        // Save consistency metrics
        hsize_t consistency_dims[1] = {static_cast<hsize_t>(num_test_runs)};
        H5::DataSpace consistency_space(1, consistency_dims);
        H5::DataSet consistency_dataset = file.createDataSet("consistency_metrics", H5::PredType::NATIVE_DOUBLE, consistency_space);
        consistency_dataset.write(test_consistency_metrics.data(), H5::PredType::NATIVE_DOUBLE);
        
        file.close();
        std::cout << "Results saved successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }
    
    // Summary statistics
    double avg_chi2 = 0.0;
    double avg_consistency = 0.0;
    for (int i = 0; i < test_chi2_values.size(); ++i) {
        avg_chi2 += test_chi2_values[i];
        avg_consistency += test_consistency_metrics[i];
    }
    avg_chi2 /= test_chi2_values.size();
    avg_consistency /= test_consistency_metrics.size();
    
    std::cout << "\n=== Nonlinear Tracking Demo Summary ===" << std::endl;
    std::cout << "Motion model: Constant Turn Rate (ω = " << turn_rate << " rad/s)" << std::endl;
    std::cout << "Measurement model: GPS Cartesian position (x, y)" << std::endl;
    std::cout << "Consistency method: " << consistency_method << std::endl;
    std::cout << "Average chi-squared: " << avg_chi2 << std::endl;
    std::cout << "Average consistency metric: " << avg_consistency << std::endl;
    std::cout << "Expected consistency metric: ≈ 1.0 (if parameters are optimal)" << std::endl;
    std::cout << "Number of test runs: " << num_test_runs << std::endl;
    std::cout << "Trajectory length: " << test_true_states[0].size() << std::endl;
    std::cout << "\nNonlinear factor graph optimization completed successfully!" << std::endl;
    
    return 0;
} 
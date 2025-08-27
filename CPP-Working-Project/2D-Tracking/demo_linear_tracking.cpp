#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "H5Cpp.h"
#include <vector>
#include <random>

// Function declaration for data generation
std::vector<std::vector<Eigen::Vector4d>> generate_test_data(const YAML::Node& config, int num_runs);
std::vector<std::vector<Eigen::Vector2d>> generate_test_measurements(const std::vector<std::vector<Eigen::Vector4d>>& states, 
                                                                     const YAML::Node& config, int num_runs);

int main() {
    std::cout << "=== Demonstrating Tuned Parameters with Custom Test Data ===" << std::endl;
    
    // Load tuned parameters from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");
    
    double tuned_q = config["validate_filter"]["q"].as<double>();
    double tuned_R = config["validate_filter"]["R"].as<double>();
    double min_objective = config["validate_filter"]["min_objective"].as<double>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    // Get the consistency method that was used in BO optimization
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>();
    
    std::cout << "Tuned Parameters:" << std::endl;
    std::cout << "  Process noise intensity (q): " << tuned_q << std::endl;
    std::cout << "  Measurement noise variance (R): " << tuned_R << std::endl;
    std::cout << "  Best CNEES objective: " << min_objective << std::endl;
    std::cout << "  Time step (dt): " << dt << std::endl;
    std::cout << "  Consistency method: " << consistency_method << std::endl;
    std::cout << std::endl;
    
    // Generate new test data with configurable parameters
    int num_demo_runs = 5;  // Number of test trajectories to generate
    
    std::cout << "Generating " << num_demo_runs << " new test trajectories..." << std::endl;
    std::cout << "Test data parameters:" << std::endl;
    std::cout << "  Trajectory length: " << config["Data_Generation"]["trajectory_length"].as<int>() << std::endl;
    std::cout << "  Initial position: [" << config["Data_Generation"]["pos"]["x"].as<double>() << ", " 
              << config["Data_Generation"]["pos"]["y"].as<double>() << "]" << std::endl;
    std::cout << "  Initial velocity: [" << config["Data_Generation"]["vel"]["x"].as<double>() << ", " 
              << config["Data_Generation"]["vel"]["y"].as<double>() << "]" << std::endl;
    std::cout << "  Process noise intensity (q): " << config["Data_Generation"]["q"].as<double>() << std::endl;
    std::cout << "  Measurement noise variance (R): " << config["Data_Generation"]["meas_noise_var"].as<double>() << std::endl;
    std::cout << "  Seed: " << config["Data_Generation"]["seed"].as<unsigned int>() << std::endl;
    std::cout << "  Use process noise: " << (config["Data_Generation"]["use_process_noise"].as<bool>() ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Generate test data
    auto all_states = generate_test_data(config, num_demo_runs);
    auto all_measurements = generate_test_measurements(all_states, config, num_demo_runs);
    
    int T = all_states[0].size();  // Trajectory length
    
    std::cout << "Running factor graph optimization on " << num_demo_runs << " test trajectories..." << std::endl;
    
    // Prepare data structures for saving results
    std::vector<std::vector<Eigen::Vector4d>> demo_true_states;
    std::vector<std::vector<Eigen::Vector2d>> demo_measurements;
    std::vector<std::vector<Eigen::Vector4d>> demo_estimates;
    std::vector<double> demo_chi2_values;
    std::vector<double> demo_consistency_metrics;
    
    for (int run = 0; run < num_demo_runs; ++run) {
        std::cout << "Processing run " << (run + 1) << "/" << num_demo_runs << "..." << std::endl;
        
        // Create and configure factor graph
        FactorGraph2DTrajectory fg;
        
        // Set Q matrix using tuned process noise intensity
        fg.setQFromProcessNoiseIntensity(tuned_q, dt);
        
        // Set R matrix using tuned measurement noise variance
        double meas_noise_std = std::sqrt(tuned_R);
        fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
        
        // Configure output options
        FactorGraph2DTrajectory::OutputOptions opts;
        opts.output_estimated_state = true;
        opts.output_true_state = true;
        opts.output_information_matrix = true;
        fg.setOutputOptions(opts);
        
        // Run optimization with correct consistency method
        if (consistency_method == "nis3") {
            fg.run(all_states[run], &all_measurements[run], dt, false);  // No optimization for NIS3
        } else {
            // Read optimizer settings
            int max_iters = config["optimizer"]["max_iterations"].as<int>(100);
            bool verbose = config["optimizer"]["verbose"].as<bool>(false);
            std::string init_mode = config["optimizer"]["init_mode"].as<std::string>("measurement");
            double pos_std = config["optimizer"]["init_jitter"]["pos_std"].as<double>(0.05);
            double vel_std = config["optimizer"]["init_jitter"]["vel_std"].as<double>(0.2);
            fg.setMaxIterations(max_iters);
            fg.setVerbose(verbose);
            fg.setInitMode(init_mode);
            fg.setInitJitter(pos_std, vel_std);
            fg.run(all_states[run], &all_measurements[run], dt, true);   // Optimization for NIS4
        }
        
        // Get results
        auto est_states = fg.getAllEstimates();
        auto true_states = fg.getAllTrueStates();
        double chi2 = fg.getChi2();
        
        {
            int breakdown_runs = config["logging"]["breakdown_runs"].as<int>(0);
            if (run < breakdown_runs) {
                auto br = fg.computeChi2Breakdown();
                std::cout << "Demo run chi2 breakdown: process=" << br.processChi2
                          << ", meas=" << br.measurementChi2 << ", total=" << br.totalChi2 << std::endl;
            }
        }
        
        // Calculate consistency metric (same as used in BO optimization)
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
        demo_true_states.push_back(true_states);
        demo_measurements.push_back(all_measurements[run]);
        demo_estimates.push_back(est_states);
        demo_chi2_values.push_back(chi2);
        demo_consistency_metrics.push_back(consistency_metric);
        
        // Print some statistics for this run
        double mean_error = 0.0;
        for (int k = 0; k < T; ++k) {
            Eigen::Vector4d error = true_states[k] - est_states[k];
            mean_error += error.head<2>().norm();  // Position error only
        }
        mean_error /= T;
        
        std::cout << "  Run " << (run + 1) << " - Chi2: " << chi2 
                  << ", Consistency Metric: " << consistency_metric
                  << ", Mean position error: " << mean_error << std::endl;
    }
    
    std::cout << "\nSaving results to HDF5..." << std::endl;
    
    // Save results to HDF5 file
    const std::string output_file = "../2D-Tracking/Saved_Data/2D_tuned_demo_results.h5";
    
    try {
        H5::H5File file(output_file, H5F_ACC_TRUNC);
        
        // Dimensions: [num_runs, trajectory_length, state_dim/meas_dim]
        hsize_t dims_states[3] = {(hsize_t)num_demo_runs, (hsize_t)T, 4};
        hsize_t dims_measurements[3] = {(hsize_t)num_demo_runs, (hsize_t)T, 2};
        hsize_t dims_chi2[1] = {(hsize_t)num_demo_runs};
        
        // Create datasets
        H5::DataSpace space_states(3, dims_states);
        H5::DataSpace space_measurements(3, dims_measurements);
        H5::DataSpace space_chi2(1, dims_chi2);
        
        H5::DataSet dataset_true = file.createDataSet("true_states", H5::PredType::NATIVE_DOUBLE, space_states);
        H5::DataSet dataset_estimates = file.createDataSet("estimated_states", H5::PredType::NATIVE_DOUBLE, space_states);
        H5::DataSet dataset_measurements = file.createDataSet("measurements", H5::PredType::NATIVE_DOUBLE, space_measurements);
        H5::DataSet dataset_chi2 = file.createDataSet("chi2_values", H5::PredType::NATIVE_DOUBLE, space_chi2);
        
        // Flatten data for HDF5 storage
        std::vector<double> flat_true_states, flat_estimates, flat_measurements;
        flat_true_states.reserve(num_demo_runs * T * 4);
        flat_estimates.reserve(num_demo_runs * T * 4);
        flat_measurements.reserve(num_demo_runs * T * 2);
        
        for (int run = 0; run < num_demo_runs; ++run) {
            for (int k = 0; k < T; ++k) {
                // True states
                for (int i = 0; i < 4; ++i) {
                    flat_true_states.push_back(demo_true_states[run][k][i]);
                }
                // Estimates
                for (int i = 0; i < 4; ++i) {
                    flat_estimates.push_back(demo_estimates[run][k][i]);
                }
                // Measurements
                for (int i = 0; i < 2; ++i) {
                    flat_measurements.push_back(demo_measurements[run][k][i]);
                }
            }
        }
        
        // Write data
        dataset_true.write(flat_true_states.data(), H5::PredType::NATIVE_DOUBLE);
        dataset_estimates.write(flat_estimates.data(), H5::PredType::NATIVE_DOUBLE);
        dataset_measurements.write(flat_measurements.data(), H5::PredType::NATIVE_DOUBLE);
        dataset_chi2.write(demo_chi2_values.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save consistency metrics
        H5::DataSet dataset_consistency = file.createDataSet("consistency_metrics", H5::PredType::NATIVE_DOUBLE, space_chi2);
        dataset_consistency.write(demo_consistency_metrics.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Add metadata as attributes
        H5::Attribute attr_q = file.createAttribute("tuned_q", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        attr_q.write(H5::PredType::NATIVE_DOUBLE, &tuned_q);
        
        H5::Attribute attr_R = file.createAttribute("tuned_R", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        attr_R.write(H5::PredType::NATIVE_DOUBLE, &tuned_R);
        
        H5::Attribute attr_dt = file.createAttribute("dt", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        attr_dt.write(H5::PredType::NATIVE_DOUBLE, &dt);
        
        H5::Attribute attr_cnees = file.createAttribute("best_cnees", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        attr_cnees.write(H5::PredType::NATIVE_DOUBLE, &min_objective);
        
        std::cout << "Results saved to: " << output_file << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "Error saving HDF5 file: " << e.getDetailMsg() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "Next step: Run 'python plot_tuned_demo.py' to visualize the results" << std::endl;
    
    // Calculate and display consistency metrics summary
    double avg_chi2 = 0.0;
    double avg_consistency = 0.0;
    for (int i = 0; i < demo_chi2_values.size(); ++i) {
        avg_chi2 += demo_chi2_values[i];
        avg_consistency += demo_consistency_metrics[i];
    }
    avg_chi2 /= demo_chi2_values.size();
    avg_consistency /= demo_consistency_metrics.size();
    
    std::cout << "\n=== Consistency Metrics Summary ===" << std::endl;
    std::cout << "Average chi-squared: " << avg_chi2 << std::endl;
    std::cout << "Average consistency metric: " << avg_consistency << std::endl;
    std::cout << "Consistency method used: " << consistency_method << std::endl;
    std::cout << "Expected consistency metric: ≈ 1.0 (if parameters are optimal)" << std::endl;
    if (avg_consistency > 1.5) {
        std::cout << "⚠️  High consistency metric suggests parameters may be too aggressive" << std::endl;
    } else if (avg_consistency < 0.5) {
        std::cout << "⚠️  Low consistency metric suggests parameters may be too conservative" << std::endl;
    } else {
        std::cout << "✅ Consistency metric looks good!" << std::endl;
    }
    
    return 0;
}

// Data generation function
std::vector<std::vector<Eigen::Vector4d>> generate_test_data(const YAML::Node& config, int num_runs) {
    // Extract parameters from config
    int N = config["Data_Generation"]["trajectory_length"].as<int>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    Eigen::Vector2d pos(config["Data_Generation"]["pos"]["x"].as<double>(), 
                       config["Data_Generation"]["pos"]["y"].as<double>());
    Eigen::Vector2d vel(config["Data_Generation"]["vel"]["x"].as<double>(), 
                       config["Data_Generation"]["vel"]["y"].as<double>());
    unsigned int base_seed = config["Data_Generation"]["seed"].as<unsigned int>();
    double V0 = config["Data_Generation"]["q"].as<double>();
    double V1 = config["Data_Generation"]["q"].as<double>();
    bool use_process_noise = config["Data_Generation"]["use_process_noise"].as<bool>();
    
    // State transition matrix F (constant velocity model)
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  // x position += x velocity * dt
    F(1, 3) = dt;  // y position += y velocity * dt
    
    // Control input matrix B
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;
    
    // Construct process noise covariance matrix Q
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt3 = dt2 * dt;
    
    if (use_process_noise) {
        // Position-position covariance (diagonal)
        Q(0, 0) = dt3 / 3.0 * V0;
        Q(1, 1) = dt3 / 3.0 * V1;
        
        // Velocity-velocity covariance (diagonal)
        Q(2, 2) = dt * V0;
        Q(3, 3) = dt * V1;
        
        // Position-velocity cross covariance
        Q(0, 2) = dt2 / 2.0 * V0;
        Q(2, 0) = Q(0, 2);
        Q(1, 3) = dt2 / 2.0 * V1;
        Q(3, 1) = Q(1, 3);
    }
    
    // Validate Q matrix
    Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q);
    if (lltOfQ.info() != Eigen::Success) {
        std::cerr << "ERROR: Q matrix is not positive semi-definite!" << std::endl;
        throw std::runtime_error("Invalid Q matrix");
    }
    
    // Generate trajectories
    std::vector<std::vector<Eigen::Vector4d>> all_states(num_runs);
    
    for (int run = 0; run < num_runs; ++run) {
        std::vector<Eigen::Vector4d> states(N);
        states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        
        std::mt19937 gen(base_seed + run); // Different seed for each run
        Eigen::Matrix4d L = lltOfQ.matrixL();
        
        for (int k = 1; k < N; ++k) {
            // State equation: xₖ₊₁ = Fxₖ + Buₖ + vₖ (with uₖ = 0 for constant velocity)
            states[k] = F * states[k-1];
            
            // Apply process noise if enabled
            if (use_process_noise) {
                // Generate uncorrelated standard normal noise
                Eigen::Vector4d uncorrelated_noise;
                std::normal_distribution<> normal_dist(0.0, 1.0);
                for (int i = 0; i < 4; ++i) {
                    uncorrelated_noise[i] = normal_dist(gen);
                }
                
                // Transform to correlated noise using Q = L*L^T
                Eigen::Vector4d process_noise = L * uncorrelated_noise;
                states[k] += process_noise;
            }
        }
        
        all_states[run] = states;
    }
    
    return all_states;
}

// Measurement generation function
std::vector<std::vector<Eigen::Vector2d>> generate_test_measurements(const std::vector<std::vector<Eigen::Vector4d>>& states, 
                                                                     const YAML::Node& config, int num_runs) {
    int N = states[0].size();
    double meas_noise_var = config["Data_Generation"]["meas_noise_var"].as<double>();
    unsigned int base_seed = config["Data_Generation"]["seed"].as<unsigned int>();
    
    // Construct measurement noise covariance matrix R
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = meas_noise_var;
    R(1, 1) = meas_noise_var;
    
    // Validate R matrix
    Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) {
        std::cerr << "ERROR: R matrix is not positive semi-definite!" << std::endl;
        throw std::runtime_error("Invalid R matrix");
    }
    
    // Generate measurements
    std::vector<std::vector<Eigen::Vector2d>> all_measurements(num_runs);
    Eigen::Matrix2d L_R = lltOfR.matrixL();
    
    for (int run = 0; run < num_runs; ++run) {
        std::vector<Eigen::Vector2d> measurements(N);
        std::mt19937 gen(base_seed + run);
        
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
            measurements[k] = states[run][k].head<2>() + measurement_noise;
        }
        
        all_measurements[run] = measurements;
    }
    
    return all_measurements;
} 
#include "fg_class_tracking.h"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>

// Function declaration for control input generation
Eigen::Vector2d generateControlInput(int time_step, double dt);

// Function to generate validation data with specified seed and parameters
void generateValidationData(
    std::vector<std::vector<Eigen::Vector4d>>& all_states,
    std::vector<std::vector<Eigen::Vector2d>>& all_measurements,
    int num_graphs, int trajectory_length,
    double dt, double process_noise_intensity, double meas_noise_std,
    const Eigen::Vector4d& initial_state, unsigned int base_seed) {
    
    std::cout << "Generating validation data with:" << std::endl;
    std::cout << "  Seed: " << base_seed << std::endl;
    std::cout << "  Process noise intensity (V0): " << process_noise_intensity << std::endl;
    std::cout << "  Measurement noise std: " << meas_noise_std << std::endl;
    std::cout << "  Number of runs: " << num_graphs << std::endl;
    std::cout << "  Trajectory length: " << trajectory_length << std::endl;
    
    // State transition matrix F (constant velocity model)
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;  // x position += x velocity * dt
    F(1, 3) = dt;  // y position += y velocity * dt
    
    // Control input matrix B (for future acceleration control)
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;
    
    // Construct the process noise covariance matrix Q
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt3 = dt2 * dt;
    
    // Position-position covariance (diagonal)
    Q(0, 0) = dt3 / 3.0 * process_noise_intensity;  // x position variance
    Q(1, 1) = dt3 / 3.0 * process_noise_intensity;  // y position variance
    
    // Velocity-velocity covariance (diagonal)
    Q(2, 2) = dt * process_noise_intensity;         // x velocity variance
    Q(3, 3) = dt * process_noise_intensity;         // y velocity variance
    
    // Position-velocity cross covariance
    Q(0, 2) = dt2 / 2.0 * process_noise_intensity;  // x position - x velocity covariance
    Q(2, 0) = Q(0, 2);                              // symmetric
    Q(1, 3) = dt2 / 2.0 * process_noise_intensity;  // y position - y velocity covariance
    Q(3, 1) = Q(1, 3);                              // symmetric
    
    // Construct the measurement noise covariance matrix R
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = meas_noise_std * meas_noise_std;  // x measurement variance
    R(1, 1) = meas_noise_std * meas_noise_std;  // y measurement variance
    
    // Validate that Q and R are positive definite
    Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q);
    if (lltOfQ.info() != Eigen::Success) {
        throw std::runtime_error("Q matrix is not positive definite!");
    }
    
    Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
    if (lltOfR.info() != Eigen::Success) {
        throw std::runtime_error("R matrix is not positive definite!");
    }
    
    std::cout << "Q and R matrices are positive definite ✓" << std::endl;
    
    // Prepare output containers
    all_states.resize(num_graphs, std::vector<Eigen::Vector4d>(trajectory_length));
    all_measurements.resize(num_graphs, std::vector<Eigen::Vector2d>(trajectory_length));
    
    // Generate correlated noise using Cholesky decomposition
    Eigen::Matrix4d L_Q = lltOfQ.matrixL();
    Eigen::Matrix2d L_R = lltOfR.matrixL();
    
    // Generate data for each run
    for (int run = 0; run < num_graphs; ++run) {
        // Use different seed for each run for reproducible but varied data
        std::mt19937 gen(base_seed + run);
        std::normal_distribution<> normal_dist(0.0, 1.0);
        
        // Initialize trajectory with initial state
        std::vector<Eigen::Vector4d> true_states(trajectory_length);
        true_states[0] = initial_state;
        
        // Generate trajectory with process noise
        for (int k = 1; k < trajectory_length; ++k) {
            // Generate control input (acceleration) - currently zero for constant velocity
            Eigen::Vector2d acceleration = generateControlInput(k, dt);
            Eigen::Vector4d control_effect = B * acceleration;
            
            // Predict next state: x_{k+1} = F * x_k + B * u_k
            true_states[k] = F * true_states[k-1] + control_effect;
            
            // Add process noise: x_{k+1} += L_Q * w_k where w_k ~ N(0,I)
            Eigen::Vector4d uncorrelated_noise;
            for (int i = 0; i < 4; ++i) {
                uncorrelated_noise[i] = normal_dist(gen);
            }
            Eigen::Vector4d process_noise = L_Q * uncorrelated_noise;
            true_states[k] += process_noise;
        }
        
        // Generate measurements with measurement noise
        std::vector<Eigen::Vector2d> measurements(trajectory_length);
        for (int k = 0; k < trajectory_length; ++k) {
            // Extract position from state
            Eigen::Vector2d true_position = true_states[k].head<2>();
            
            // Add measurement noise: z_k = H * x_k + L_R * v_k where v_k ~ N(0,I)
            Eigen::Vector2d uncorrelated_meas_noise;
            for (int i = 0; i < 2; ++i) {
                uncorrelated_meas_noise[i] = normal_dist(gen);
            }
            Eigen::Vector2d measurement_noise = L_R * uncorrelated_meas_noise;
            measurements[k] = true_position + measurement_noise;
        }
        
        // Store results
        all_states[run] = true_states;
        all_measurements[run] = measurements;
    }
    
    std::cout << "Validation data generation complete." << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Q and R Matrix Validation Tool ===" << std::endl;
    std::cout << "This tool validates filter performance using configurable Q and R values" << std::endl;
    std::cout << "from the 'validate_filter' section of BO_Parameters.yaml" << std::endl;
    
    // Parse command line arguments for optional seed override
    unsigned int validation_seed = 12345;  // Default validation seed
    if (argc > 1) {
        try {
            validation_seed = std::stoul(argv[1]);
            std::cout << "Using custom seed: " << validation_seed << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Invalid seed argument, using default: " << validation_seed << std::endl;
        }
    } else {
        std::cout << "Using default seed: " << validation_seed << std::endl;
        std::cout << "Usage: ./Validate_QR [seed] to specify custom seed" << std::endl;
    }
    
    // Load configuration from YAML file
    YAML::Node config;
    try {
        config = YAML::LoadFile("../BO_Parameters.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return 1;
    }
    
    // Read TRUE parameters (for data generation - same as original BO training data)
    double true_V0 = config["Data_Generation"]["q"].as<double>();
    double true_R_var = config["Data_Generation"]["meas_noise_var"].as<double>();
    double true_meas_noise_std = std::sqrt(true_R_var);
    
    // Read OPTIMIZED parameters (found by BO - to be tested against true parameters)
    double optimized_V0 = config["validate_filter"]["q"].as<double>();
    double optimized_R_var = config["validate_filter"]["R"].as<double>();
    double optimized_meas_noise_std = std::sqrt(optimized_R_var);
    
    // Read data generation parameters
    double dt = config["Data_Generation"]["dt"].as<double>();
    int num_graphs = config["Data_Generation"]["num_graphs"].as<int>();
    int trajectory_length = config["Data_Generation"]["trajectory_length"].as<int>();
    
    // Read initial conditions
    Eigen::Vector4d initial_state;
    initial_state << 
        config["Data_Generation"]["pos"]["x"].as<double>(),
        config["Data_Generation"]["pos"]["y"].as<double>(),
        config["Data_Generation"]["vel"]["x"].as<double>(),
        config["Data_Generation"]["vel"]["y"].as<double>();
    
    // Read consistency method
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>("nees");
    
    std::cout << "\n=== Validation Configuration ===" << std::endl;
    std::cout << "TRUE parameters (for data generation):" << std::endl;
    std::cout << "  Q (process noise intensity): " << true_V0 << std::endl;
    std::cout << "  R (measurement noise var): " << true_R_var << std::endl;
    std::cout << "  Measurement noise std: " << true_meas_noise_std << std::endl;
    std::cout << "OPTIMIZED parameters (found by BO, to be tested):" << std::endl;
    std::cout << "  Q (process noise intensity): " << optimized_V0 << std::endl;
    std::cout << "  R (measurement noise var): " << optimized_R_var << std::endl;
    std::cout << "  Measurement noise std: " << optimized_meas_noise_std << std::endl;
    std::cout << "Time step (dt): " << dt << std::endl;
    std::cout << "Number of Monte Carlo runs: " << num_graphs << std::endl;
    std::cout << "Trajectory length: " << trajectory_length << std::endl;
    std::cout << "Consistency method: " << consistency_method << std::endl;
    std::cout << "Initial state: [" << initial_state.transpose() << "]" << std::endl;
    
    // Generate validation data using validation parameters
    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    
    std::cout << "\n=== Data Generation ===" << std::endl;
    std::cout << "Generating data with TRUE parameters (q=" << true_V0 << ", R=" << true_R_var << ")..." << std::endl;
    try {
        generateValidationData(
            all_states, all_measurements,
            num_graphs, trajectory_length,
            dt, true_V0, true_meas_noise_std,
            initial_state, validation_seed
        );
    } catch (const std::exception& e) {
        std::cerr << "Error during data generation: " << e.what() << std::endl;
        return 1;
    }
    
    // Run validation using OPTIMIZED parameters for the filter
    std::cout << "\n=== Running Validation ===" << std::endl;
    std::cout << "Testing filter with OPTIMIZED parameters (q=" << optimized_V0 << ", R=" << optimized_R_var << ")..." << std::endl;
    std::cout << "If optimized ≈ true, expect good results. If optimized ≠ true, expect poor results." << std::endl;
    
    std::vector<double> consistency_values(num_graphs, 0.0);
    
    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        
        // CRITICAL: Use OPTIMIZED parameters for the filter (testing how good BO results are)
        fg.setQFromProcessNoiseIntensity(optimized_V0, dt);
        fg.setRFromMeasurementNoise(optimized_meas_noise_std, optimized_meas_noise_std);
        
        double consistency_value = 0.0;
        
        if (consistency_method == "nis3" || consistency_method == "nis4") {
            // Use MATLAB student approach for NIS3/NIS4
            bool do_optimization = (consistency_method == "nis4");
            fg.run(all_states[run], &all_measurements[run], dt, do_optimization);
            consistency_value = fg.getChi2();
        } else {
            // NEES calculation
            fg.run(all_states[run], &all_measurements[run], dt, true);  // Always optimize for NEES
            
            auto est_states = fg.getAllEstimates();
            Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
            
            if (Hessian.rows() == 0) {
                std::cerr << "Warning: Empty Hessian matrix for run " << run << std::endl;
                consistency_value = 0.0;
            } else {
                // Calculate NEES: (x_true - x_est)^T * H * (x_true - x_est)
                Eigen::VectorXd full_error(trajectory_length * 4);
                for (int k = 0; k < trajectory_length; ++k) {
                    Eigen::Vector4d err = all_states[run][k] - est_states[k];
                    full_error.segment<4>(k * 4) = err;
                }
                consistency_value = full_error.transpose() * Hessian * full_error;
            }
        }
        
        consistency_values[run] = consistency_value;
        
        // Progress reporting
        if (run == 0) {
            std::cout << "First run " << consistency_method << " value: " << consistency_value << std::endl;
        }
    }
    
    // Calculate statistics
    std::cout << "\n=== Results Analysis ===" << std::endl;
    
    double sample_mean = 0.0;
    for (double value : consistency_values) {
        sample_mean += value;
    }
    sample_mean /= num_graphs;
    
    double sample_variance = 0.0;
    for (double value : consistency_values) {
        sample_variance += (value - sample_mean) * (value - sample_mean);
    }
    if (num_graphs > 1) {
        sample_variance /= (num_graphs - 1);
    }
    
    // Calculate degrees of freedom using actual graph dimensions
    FactorGraph2DTrajectory temp_fg;
    temp_fg.setQFromProcessNoiseIntensity(optimized_V0, dt);
    temp_fg.setRFromMeasurementNoise(optimized_meas_noise_std, optimized_meas_noise_std);
    temp_fg.run(all_states[0], &all_measurements[0], dt, true);
    
    auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();
    
    int total_dof;
    if (consistency_method == "nis3") {
        total_dof = dimZ;  // Proposition 3: DOF = total edge dimensions
        std::cout << "NIS3 (Proposition 3): DOF = dimZ = " << total_dof << std::endl;
    } else if (consistency_method == "nis4") {
        total_dof = dimZ - dimX;  // Proposition 4: DOF = edge dims - vertex dims
        std::cout << "NIS4 (Proposition 4): DOF = dimZ - dimX = " << dimZ << " - " << dimX << " = " << total_dof << std::endl;
    } else {
        total_dof = dimX;  // NEES: DOF = total vertex dimensions
        std::cout << "NEES: DOF = dimX = " << total_dof << std::endl;
    }
    
    double theoretical_mean = total_dof;
    double theoretical_variance = 2 * total_dof;
    
    std::cout << "\n=== " << consistency_method << " Validation Results ===" << std::endl;
    std::cout << "Parameters used:" << std::endl;
    std::cout << "  Q (process noise intensity): " << optimized_V0 << std::endl;
    std::cout << "  R (measurement noise var): " << optimized_R_var << std::endl;
    std::cout << "  Seed: " << validation_seed << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "              |  Observed       |  Expected (χ², k=" << total_dof << ")" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    printf("Mean          | %12.4f    | %12.4f\n", sample_mean, theoretical_mean);
    printf("Variance      | %12.4f    | %12.4f\n", sample_variance, theoretical_variance);
    std::cout << "---------------------------------" << std::endl;
    
    // Calculate consistency metric
    double log_mean = std::log(sample_mean / total_dof);
    double log_variance = (sample_variance > 0) ? std::log(sample_variance / (2.0 * total_dof)) : 0.0;
    double consistency_metric = std::abs(log_mean) + std::abs(log_variance);
    
    double normalized_mean = sample_mean / total_dof;
    std::cout << "Normalized mean: " << normalized_mean << " (should be ~1.0 for consistency)" << std::endl;
    std::cout << "Consistency metric: " << consistency_metric << " (lower is better)" << std::endl;
    
    // Save results
    try {
        std::string h5_filename = "../2D-Tracking/Saved_Data/2D_" + consistency_method + "_validation_seed_" + std::to_string(validation_seed) + ".h5";
        
        H5::H5File file(h5_filename, H5F_ACC_TRUNC);
        
        // Save consistency values
        hsize_t dims[1] = {static_cast<hsize_t>(num_graphs)};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = file.createDataSet(consistency_method + "_values", H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(consistency_values.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save metadata as attributes
        H5::DataSpace attr_space(H5S_SCALAR);
        auto createDoubleAttr = [&](const std::string& name, double value) {
            H5::Attribute attr = file.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &value);
        };
        auto createIntAttr = [&](const std::string& name, int value) {
            H5::Attribute attr = file.createAttribute(name, H5::PredType::NATIVE_INT, attr_space);
            attr.write(H5::PredType::NATIVE_INT, &value);
        };
        
        createDoubleAttr("true_V0", true_V0);
        createDoubleAttr("true_R_var", true_R_var);
        createDoubleAttr("optimized_V0", optimized_V0);
        createDoubleAttr("optimized_R_var", optimized_R_var);
        createDoubleAttr("sample_mean", sample_mean);
        createDoubleAttr("sample_variance", sample_variance);
        createDoubleAttr("theoretical_mean", theoretical_mean);
        createDoubleAttr("theoretical_variance", theoretical_variance);
        createDoubleAttr("consistency_metric", consistency_metric);
        createIntAttr("validation_seed", static_cast<int>(validation_seed));
        createIntAttr("total_dof", total_dof);
        createIntAttr("num_graphs", num_graphs);
        createIntAttr("trajectory_length", trajectory_length);
        
        file.close();
        std::cout << "\nResults saved to: " << h5_filename << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "Error saving results: " << e.getDetailMsg() << std::endl;
    }
    
    std::cout << "\n=== Validation Complete ===" << std::endl;
    return 0;
}

// Control input generation function - currently returns zero for constant velocity
Eigen::Vector2d generateControlInput(int time_step, double dt) {
    // For constant velocity model, return zero acceleration
    return Eigen::Vector2d::Zero();
} 
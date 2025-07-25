#include "fg_class_tracking.h"
#include "2D_h5_loader.h"  // Add this include
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

// Function declaration for control input generation
Eigen::Vector2d generateControlInput(int time_step, double dt);

// Function to read the best Q and R values from the optimization results file
bool load_best_params(double& best_V0, double& best_meas_noise_var) {
    std::ifstream infile("../2D-Tracking/Saved_Data/2D_bayesopt_best.txt");
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open 2D_bayesopt_best.txt to read V0 and meas_noise_var values." << std::endl;
        std::cerr << "Please ensure the file exists and contains the best parameters." << std::endl;
        return false;
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string temp;
        char comma;

        // Expects format: "Best V0: 0.123, Best meas_noise_std²: 4.56, ..."
        if (ss >> temp >> temp >> best_V0 >> comma >> temp >> temp >> temp >> best_meas_noise_var) {
             std::cout << "Successfully loaded Best V0: " << best_V0 << " and Best meas_noise_std²: " << best_meas_noise_var << std::endl;
             return true;
        }
    }
    
    std::cerr << "Error: Could not parse V0 and meas_noise_var from 2D_bayesopt_best.txt." << std::endl;
    return false;
}

int main(int argc, char* argv[]) {
    // Check command line arguments for data mode
    bool use_existing_data = false;
    if (argc > 1 && std::string(argv[1]) == "--use-existing-data") {
        use_existing_data = true;
        std::cout << "Mode: Using existing HDF5 data (same as BO_Tracking_CNEES)" << std::endl;
    } else {
        std::cout << "Mode: Generating new validation data" << std::endl;
        std::cout << "Use --use-existing-data to validate with same data as BO_Tracking_CNEES" << std::endl;
    }

    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");

    // --- PART 1: SETUP ---

    double best_V0 = config["validate_filter"]["q"].as<double>();  // Process noise intensity
    double best_meas_noise_var = config["validate_filter"]["R"].as<double>();  // Measurement noise variance from BayesOpt
    double best_meas_noise_std = sqrt(best_meas_noise_var);  // Convert variance to std dev
    
    // if (!load_best_params(best_V0, best_meas_noise_var)) {
    //     return 1;
    // }

    // Get dt from data generation config
    double dt = config["Data_Generation"]["dt"].as<double>();
    std::cout << "Using dt = " << dt << " from data generation config" << std::endl;

    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    int num_graphs, N, nx = 4;

    if (use_existing_data) {
        // --- OPTION 1: Load the SAME data that BO_Tracking_CNEES used ---
        std::cout << "\nLoading existing HDF5 data..." << std::endl;
        all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5");
        all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_noisy_measurements.h5");
        
        if (all_states.empty() || all_measurements.empty()) {
            std::cerr << "Could not load data from HDF5 files." << std::endl;
            return 1;
        }
        
        num_graphs = all_states.size();
        N = all_states[0].size();
        std::cout << "Loaded " << num_graphs << " runs, each of length " << N << std::endl;
        
    } else {
        // --- OPTION 2: Generate NEW, INDEPENDENT validation data ---
        std::cout << "\nGenerating new validation data..." << std::endl;
        
        // Parameters for new data generation
        N = config["Data_Generation"]["trajectory_length"].as<int>();
        num_graphs = config["Data_Generation"]["num_graphs"].as<int>();
        Eigen::Vector2d pos(config["Data_Generation"]["pos"]["x"].as<double>(), config["Data_Generation"]["pos"]["y"].as<double>());
        Eigen::Vector2d vel(config["Data_Generation"]["vel"]["x"].as<double>(), config["Data_Generation"]["vel"]["y"].as<double>());
        unsigned int base_seed = 42; // Different seed for independent validation data

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

        // Construct the process noise covariance matrix Q for 2D linear tracking
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        double dt3 = dt2 * dt;
        
        // Position-position covariance (diagonal)
        Q(0, 0) = dt3 / 3.0 * best_V0;  // x position variance
        Q(1, 1) = dt3 / 3.0 * best_V0;  // y position variance (same as V0 for now)
        
        // Velocity-velocity covariance (diagonal)
        Q(2, 2) = dt * best_V0;         // x velocity variance
        Q(3, 3) = dt * best_V0;         // y velocity variance
        
        // Position-velocity cross covariance
        Q(0, 2) = dt2 / 2.0 * best_V0;  // x position - x velocity covariance
        Q(2, 0) = Q(0, 2);              // symmetric
        Q(1, 3) = dt2 / 2.0 * best_V0;  // y position - y velocity covariance
        Q(3, 1) = Q(1, 3);              // symmetric

        // Construct the measurement noise covariance matrix R
        Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
        R(0, 0) = best_meas_noise_std * best_meas_noise_std;  // x measurement variance
        R(1, 1) = best_meas_noise_std * best_meas_noise_std;  // y measurement variance

        // Validate that Q and R are positive semi-definite
        Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q);
        if (lltOfQ.info() != Eigen::Success) {
            std::cerr << "ERROR: Q matrix is not positive semi-definite!" << std::endl;
            std::cerr << "Q matrix:" << std::endl << Q << std::endl;
            std::cerr << "V0 = " << best_V0 << ", dt = " << dt << std::endl;
            return -1;
        }

        Eigen::LLT<Eigen::Matrix2d> lltOfR(R);
        if (lltOfR.info() != Eigen::Success) {
            std::cerr << "ERROR: R matrix is not positive semi-definite!" << std::endl;
            std::cerr << "R matrix:" << std::endl << R << std::endl;
            std::cerr << "meas_noise_std = " << best_meas_noise_std << std::endl;
            return -1;
        }

        std::cout << "Q and R matrices are positive semi-definite ✓" << std::endl;

        // Generate new data in memory
        std::cout << "Generating " << num_graphs << " new trajectories for validation..." << std::endl;

        all_states.resize(num_graphs, std::vector<Eigen::Vector4d>(N));
        all_measurements.resize(num_graphs, std::vector<Eigen::Vector2d>(N));

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
                Eigen::Vector2d acceleration = generateControlInput(k, dt);  // Currently returns zero
                
                // State equation: xₖ₊₁ = Fxₖ + Buₖ + vₖ
                Eigen::Vector4d control_effect = B * acceleration;
                true_states[k] = F * true_states[k-1] + control_effect;
                
                // Generate uncorrelated standard normal noise
                Eigen::Vector4d uncorrelated_noise;
                std::normal_distribution<> normal_dist(0.0, 1.0);
                for (int i = 0; i < 4; ++i) {
                    uncorrelated_noise[i] = normal_dist(gen);
                }
                
                // Transform to correlated noise using Q = L*L^T
                Eigen::Vector4d process_noise = L * uncorrelated_noise;
                true_states[k] += process_noise;
                
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
                all_states[run][k] = true_states[k];
                all_measurements[run][k] = noisy_measurements[k];
            }
        }
        
        std::cout << "Data generation complete." << std::endl;
    }

    std::cout << "Using V0 = " << best_V0 << ", meas_noise_std = " << best_meas_noise_std << ", dt = " << dt << std::endl;

    // --- PART 2: RUN FILTER AND CALCULATE NEES ---

    if (use_existing_data) {
        std::cout << "\nRunning filter with V0=" << best_V0 << " and meas_noise_std=" << best_meas_noise_std << " on SAME data as BO..." << std::endl;
    } else {
        std::cout << "\nRunning filter with V0=" << best_V0 << " and meas_noise_std=" << best_meas_noise_std << " on NEW validation data..." << std::endl;
    }

    // Initialize accumulators for NEES statistics - following paper's approach
    // Store NEES values for each run and time step: nees[run][time_step]
    std::vector<std::vector<double>> nees_per_run_per_time(num_graphs, std::vector<double>(N, 0.0));

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        
        // Use proper Q matrix structure for 2D linear tracking with theoretical structure
        fg.setQFromProcessNoiseIntensity(best_V0, dt);
        
        // Use proper R matrix structure (2x2 diagonal matrix)
        fg.setRFromMeasurementNoise(best_meas_noise_std, best_meas_noise_std);
        
        // This run uses either the same data as BO or new validation data
        fg.run(all_states[run], &all_measurements[run], false, dt);

        auto est_states = fg.getAllEstimates();
        Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
        Eigen::MatrixXd full_cov = Hessian.inverse();
        
        // Debug: Check if information matrix looks reasonable
        if (run == 0) {
            std::cout << "Debug: Hessian matrix size: " << Hessian.rows() << "x" << Hessian.cols() << std::endl;
            std::cout << "Debug: First 4x4 block of Hessian matrix:" << std::endl;
            std::cout << Hessian.block<4,4>(0,0) << std::endl;
            
            // Check a few more blocks to see the pattern
            std::cout << "Debug: Middle 4x4 block (k=10):" << std::endl;
            std::cout << Hessian.block<4,4>(10*4,10*4) << std::endl;
            std::cout << "Debug: Last 4x4 block (k=19):" << std::endl;
            std::cout << Hessian.block<4,4>(19*4,19*4) << std::endl;
            
            // Check process model contribution by looking at off-diagonal blocks
            std::cout << "Debug: Process model blocks (k=0,1):" << std::endl;
            std::cout << Hessian.block<4,4>(0*4,1*4) << std::endl;
            std::cout << "Debug: Process model blocks (k=10,11):" << std::endl;
            std::cout << Hessian.block<4,4>(10*4,11*4) << std::endl;
        }
        
        for (int k = 0; k < N; ++k) {
            Eigen::Vector4d err = all_states[run][k] - est_states[k];
            Eigen::Matrix4d P_k_inv = full_cov.block<4,4>(k*4, k*4);
            
            // NEES = err^T * P^(-1) * err = err^T * H * err (where H is the information matrix)
            double nees_k = err.transpose() * P_k_inv * err;
            
            // Store NEES value for this run and time step
            nees_per_run_per_time[run][k] = nees_k;
            
            // Debug: Track error magnitudes and info matrix traces for first run
            if (run == 0) {
                double err_magnitude = err.norm();
                double P_ktrace = full_cov.block<4,4>(k*4, k*4).trace();
                printf("k=%d: err_mag=%.3f, P_ktrace=%.3f, NEES=%.3f\n", 
                       k, err_magnitude, P_ktrace, nees_k);
            }
        }
    }

    // --- PART 3: ANALYZE AND REPORT RESULTS (Following Paper's Approach) ---
    
    std::cout << "\nValidation complete. Analyzing results..." << std::endl;

    // Step 1: Average NEES across runs at each time step (Equation 20 from paper)
    // ε̄_{x,k} = (1/N) Σ^N_{i=1} ε^i_{x,k}
    std::vector<double> mean_nees_per_timestep(N, 0.0);
    for (int k = 0; k < N; ++k) {
        for (int run = 0; run < num_graphs; ++run) {
            mean_nees_per_timestep[k] += nees_per_run_per_time[run][k];
        }
        mean_nees_per_timestep[k] /= num_graphs;  // Average across runs
    }

    // Step 2: Sum averaged NEES across all time steps (part of Equation 36 from paper)
    // Σ^T_{k=1} ε̄_{x,k}
    double sum_of_mean_nees = 0.0;
    for (int k = 0; k < N; ++k) {
        sum_of_mean_nees += mean_nees_per_timestep[k];
    }

    // Calculate overall statistics following the paper's approach
    double sample_mean = sum_of_mean_nees / N;  // Average over time steps
    
    // Calculate variance of mean NEES per timestep
    double sample_variance = 0.0;
    for (int k = 0; k < N; ++k) {
        sample_variance += (mean_nees_per_timestep[k] - sample_mean) * (mean_nees_per_timestep[k] - sample_mean);
    }
    sample_variance /= (N - 1);

    // Also calculate traditional statistics for comparison
    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    int total_count = 0;
    
    for (int run = 0; run < num_graphs; ++run) {
        for (int k = 0; k < N; ++k) {
            total_sum += nees_per_run_per_time[run][k];
            total_sum_sq += nees_per_run_per_time[run][k] * nees_per_run_per_time[run][k];
            total_count++;
        }
    }
    
    double traditional_mean = total_sum / total_count;
    double traditional_variance = (total_sum_sq / total_count) - (traditional_mean * traditional_mean);

    // --- Theoretical Values for Chi-Squared with k=4 degrees of freedom ---
    double theoretical_mean = nx;
    double theoretical_variance = 2 * nx;

    std::cout << "\n--- NEES Validation Results (Following Paper's Method) ---" << std::endl;
    if (use_existing_data) {
        std::cout << "Data: Same as BO_Tracking_CNEES (consistency check)" << std::endl;
    } else {
        std::cout << "Data: New validation data (generalization test)" << std::endl;
    }
    std::cout << "Total NEES samples calculated: " << total_count << std::endl;
    std::cout << "Method: Average across runs first, then across time (Paper's Eq. 20 & 36)" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "              |  Paper Method   |  Traditional    |  Theoretical (χ², k=4)" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    printf("Mean          | %12.4f    | %12.4f    | %12.4f\n", sample_mean, traditional_mean, theoretical_mean);
    printf("Variance      | %12.4f    | %12.4f    | %12.4f\n", sample_variance, traditional_variance, theoretical_variance);
    std::cout << "---------------------------------" << std::endl;
    
    // Calculate Paper's cost function J_NEES (Equation 36)
    // y(q) = J_NEES(q) = sqrt(log((Σ^T_{k=1} ε̄_{x,k})/n_x)^2)
    double j_nees_cost = std::sqrt(std::pow(std::log(sum_of_mean_nees / nx), 2));
    std::cout << "Paper's J_NEES cost function: " << j_nees_cost << std::endl;
    
    // Calculate NEES means and variances per Monte Carlo run for HDF5 output
    std::vector<double> nees_means_per_run(num_graphs);
    std::vector<double> nees_variances_per_run(num_graphs);
    
    for (int run = 0; run < num_graphs; ++run) {
        double run_sum = 0.0;
        for (int k = 0; k < N; ++k) {
            run_sum += nees_per_run_per_time[run][k];
        }
        nees_means_per_run[run] = run_sum / N;
        
        double run_var = 0.0;
        for (int k = 0; k < N; ++k) {
            run_var += (nees_per_run_per_time[run][k] - nees_means_per_run[run]) * 
                       (nees_per_run_per_time[run][k] - nees_means_per_run[run]);
        }
        nees_variances_per_run[run] = run_var / (N - 1);
    }
    
    // Save NEES statistics to HDF5 file
    try {
        std::string h5_filename;
        if (use_existing_data) {
            h5_filename = "../2D-Tracking/Saved_Data/2D_nees_validation_same_data.h5";
        } else {
            h5_filename = "../2D-Tracking/Saved_Data/2D_nees_validation_new_data.h5";
        }
        
        H5::H5File file(h5_filename, H5F_ACC_TRUNC);
        
        // Save Monte Carlo run numbers
        hsize_t run_dims[1] = {static_cast<hsize_t>(num_graphs)};
        H5::DataSpace run_dataspace(1, run_dims);
        H5::DataSet run_dataset = file.createDataSet("monte_carlo_runs", H5::PredType::NATIVE_DOUBLE, run_dataspace);
        std::vector<double> run_numbers(num_graphs);
        for (int run = 0; run < num_graphs; ++run) {
            run_numbers[run] = static_cast<double>(run + 1);  // 1-indexed for display
        }
        run_dataset.write(run_numbers.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save NEES means per Monte Carlo run
        H5::DataSet mean_dataset = file.createDataSet("nees_means", H5::PredType::NATIVE_DOUBLE, run_dataspace);
        mean_dataset.write(nees_means_per_run.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save NEES variances per Monte Carlo run
        H5::DataSet variance_dataset = file.createDataSet("nees_variances", H5::PredType::NATIVE_DOUBLE, run_dataspace);
        variance_dataset.write(nees_variances_per_run.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save overall statistics as attributes
        H5::DataSpace attr_dataspace(H5S_SCALAR);
        H5::Attribute overall_mean_attr = file.createAttribute("overall_mean", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute overall_variance_attr = file.createAttribute("overall_variance", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute theoretical_mean_attr = file.createAttribute("theoretical_mean", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute theoretical_variance_attr = file.createAttribute("theoretical_variance", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute V0_attr = file.createAttribute("V0", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute meas_noise_std_attr = file.createAttribute("meas_noise_std", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute use_existing_data_attr = file.createAttribute("use_existing_data", H5::PredType::NATIVE_INT, attr_dataspace);
        
        overall_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_mean);
        overall_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_variance);
        theoretical_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_mean);
        theoretical_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_variance);
        V0_attr.write(H5::PredType::NATIVE_DOUBLE, &best_V0);
        meas_noise_std_attr.write(H5::PredType::NATIVE_DOUBLE, &best_meas_noise_std);
        int use_existing_flag = use_existing_data ? 1 : 0;
        use_existing_data_attr.write(H5::PredType::NATIVE_INT, &use_existing_flag);
        
        file.close();
        std::cout << "NEES validation results saved to " << h5_filename << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "Error saving to HDF5: " << e.getDetailMsg() << std::endl;
    }

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
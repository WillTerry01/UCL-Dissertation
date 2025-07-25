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
    
    // Get consistency method from YAML config
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>("nees");
    std::cout << "Consistency method: " << consistency_method << std::endl;

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
        unsigned int base_seed = 1337; // Different seed for independent validation data

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

    // --- PART 2: RUN FILTER AND CALCULATE CONSISTENCY METRICS ---

    if (use_existing_data) {
        std::cout << "\nRunning filter with V0=" << best_V0 << " and meas_noise_std=" << best_meas_noise_std << " on SAME data as BO..." << std::endl;
    } else {
        std::cout << "\nRunning filter with V0=" << best_V0 << " and meas_noise_std=" << best_meas_noise_std << " on NEW validation data..." << std::endl;
    }

    std::cout << "Computing " << consistency_method << " consistency metrics..." << std::endl;

    // Initialize accumulators for consistency statistics  
    // Store one consistency value per run: consistency_values[run]
    std::vector<double> consistency_values(num_graphs, 0.0);

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
        
        // Check if covariance matrix is valid
        if (full_cov.rows() == 0) {
            std::cerr << "ERROR: Covariance matrix is empty for run " << run << std::endl;
            consistency_values[run] = 0.0;
            continue;
        }
        
        double consistency_value = 0.0;
        
        if (consistency_method == "nis3" || consistency_method == "nis4") {
            // CNIS calculation using full trajectory approach
            
            // Measurement Jacobian H for 2D position measurements: H = [I_2x2, 0_2x2]
            Eigen::Matrix<double, 2, 4> H;
            H << 1, 0, 0, 0,    // x_pos = state[0]
                 0, 1, 0, 0;    // y_pos = state[1]
            
            // Measurement noise covariance R (2x2)
            Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (best_meas_noise_std * best_meas_noise_std);
            
            // Create full innovation vector by stacking all timestep innovations
            Eigen::VectorXd full_innovation(N * 2);
            for (int k = 0; k < N; ++k) {
                // Predicted measurement: z_pred = H * x_est
                Eigen::Vector2d z_predicted = H * est_states[k];
                
                // Actual measurement
                Eigen::Vector2d z_actual = all_measurements[run][k];
                
                // Innovation: y = z_actual - z_predicted
                Eigen::Vector2d innovation = z_actual - z_predicted;
                
                // Stack into full innovation vector
                full_innovation.segment<2>(k * 2) = innovation;
            }
            
            // Construct full measurement Jacobian H_full (N*2 x N*4)
            Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(N * 2, N * 4);
            for (int k = 0; k < N; ++k) {
                H_full.block<2,4>(k * 2, k * 4) = H;  // Place H in diagonal blocks
            }
            
            // Construct full measurement noise covariance R_full (N*2 x N*2, block diagonal)
            Eigen::MatrixXd R_full = Eigen::MatrixXd::Zero(N * 2, N * 2);
            for (int k = 0; k < N; ++k) {
                R_full.block<2,2>(k * 2, k * 2) = R;  // Place R in diagonal blocks
            }
            
            // Full innovation covariance: S_full = H_full * P_full * H_full^T + R_full
            Eigen::MatrixXd S_full = H_full * full_cov * H_full.transpose() + R_full;
            
            // Check if S_full is invertible
            Eigen::LLT<Eigen::MatrixXd> lltOfS(S_full);
            if (lltOfS.info() != Eigen::Success) {
                std::cerr << "Warning: Full innovation covariance is not positive definite for run " << run << std::endl;
                consistency_values[run] = 0.0;
                continue;
            }
            
            // Calculate full-trajectory NIS: innovation^T * S_full^(-1) * innovation
            consistency_value = full_innovation.transpose() * S_full.inverse() * full_innovation;
        } else {
            // CNEES calculation using full trajectory approach
            
            // Create full error vector by stacking all time step errors
            Eigen::VectorXd full_error(N * 4);
            for (int k = 0; k < N; ++k) {
                Eigen::Vector4d err = all_states[run][k] - est_states[k];
                full_error.segment<4>(k * 4) = err;
            }
            
            // Check if full covariance matrix is positive definite
            Eigen::LLT<Eigen::MatrixXd> lltOfCov(full_cov);
            if (lltOfCov.info() != Eigen::Success) {
                std::cerr << "Warning: Full covariance matrix is not positive definite for run " << run << std::endl;
                consistency_values[run] = 0.0;
                continue;
            }
            
            // Calculate NEES using full system: err^T * P^(-1) * err = err^T * H * err
            consistency_value = full_error.transpose() * Hessian * full_error;
        }
        
        consistency_values[run] = consistency_value;
        
        // Debug: Track properties for first run
        if (run == 0) {
            std::cout << "Run 0 " << consistency_method << " value: " << consistency_value << std::endl;
        }
    }

    // --- PART 3: ANALYZE AND REPORT RESULTS ---
    
    std::cout << "\nValidation complete. Analyzing results..." << std::endl;

    // Calculate mean and variance of consistency values across runs
    double sample_mean = 0.0;
    for (int run = 0; run < num_graphs; ++run) {
        sample_mean += consistency_values[run];
    }
    sample_mean /= num_graphs;  // Average across runs
    
    double sample_variance = 0.0;
    for (int run = 0; run < num_graphs; ++run) {
        sample_variance += (consistency_values[run] - sample_mean) * (consistency_values[run] - sample_mean);
    }
    if (num_graphs > 1) {
        sample_variance /= (num_graphs - 1);
    } else {
        sample_variance = 0.0;
    }

    // DOF calculation based on consistency method
    int nz = 2; // measurement dimension for 2D position measurements
    int total_dof;
    
    if (consistency_method == "nis3") {
        // Proposition 3: DOF = Nz (total residuals)
        int Nz = N * nz + (N - 1) * nx;
        total_dof = Nz;
        std::cout << "Using Khosoussi Proposition 3: DOF = Nz = " << total_dof << std::endl;
    } else if (consistency_method == "nis4") {
        // Proposition 4: DOF = Nz - Nx (total residuals - total state parameters)
        int Nx = N * nx;
        int Nz = N * nz + (N - 1) * nx;
        total_dof = Nz - Nx;
        std::cout << "Using Khosoussi Proposition 4: DOF = Nz - Nx = " << Nz << " - " << Nx << " = " << total_dof << std::endl;
    } else {
        // NEES: DOF = total state parameters
        total_dof = N * nx;
        std::cout << "Using NEES formulation: DOF = T × nx = " << total_dof << std::endl;
    }
    double theoretical_mean = total_dof;
    double theoretical_variance = 2 * total_dof;

    std::cout << "\n--- " << consistency_method << " Validation Results (Full System Method) ---" << std::endl;
    if (use_existing_data) {
        std::cout << "Data: Same as BO_Tracking_CNEES (consistency check)" << std::endl;
    } else {
        std::cout << "Data: New validation data (generalization test)" << std::endl;
    }
    std::cout << "Total " << consistency_method << " samples calculated: " << num_graphs << " (one per run)" << std::endl;
    std::cout << "Method: Full system covariance matrix (includes off-diagonal correlations)" << std::endl;
    
    // Show DOF comparison for different methods
    int dof_nees = N * nx;
    int Nz = N * nz + (N - 1) * nx;
    int Nx = N * nx;
    int dof_nis3 = Nz;
    int dof_nis4 = Nz - Nx;
    std::cout << "DOF Values - NEES: " << dof_nees 
              << ", NIS3: " << dof_nis3 
              << ", NIS4: " << dof_nis4 
              << ", Using: " << total_dof << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "              |  Full System    |  Theoretical (χ², k=" << total_dof << ")" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    printf("Mean          | %12.4f    | %12.4f\n", sample_mean, theoretical_mean);
    printf("Variance      | %12.4f    | %12.4f\n", sample_variance, theoretical_variance);
    std::cout << "---------------------------------" << std::endl;
    
    // Calculate consistency metric using the formula:
    // C = |log(ε̃ / n)| + |log(S̃ / (2*n))|
    double log_mean = std::log(sample_mean / total_dof);
    double log_variance = (sample_variance > 0) ? 
                         std::log(sample_variance / (2.0 * total_dof)) : 
                         0.0; // Avoid log(0)
    double consistency_metric = std::abs(log_mean) + std::abs(log_variance);
    
    double normalized_mean = sample_mean / total_dof;
    std::cout << "Normalized mean " << consistency_method << ": " << normalized_mean << " (should be ~1.0 for consistency)" << std::endl;
    std::cout << "Full System " << consistency_method << " metric: " << consistency_metric << std::endl;
    
    // Save consistency statistics to HDF5 file
    try {
        std::string h5_filename;
        if (use_existing_data) {
            h5_filename = "../2D-Tracking/Saved_Data/2D_" + consistency_method + "_validation_same_data.h5";
        } else {
            h5_filename = "../2D-Tracking/Saved_Data/2D_" + consistency_method + "_validation_new_data.h5";
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
        
        // Save full system consistency values per Monte Carlo run
        std::string dataset_name = consistency_method + "_full_system_values";
        H5::DataSet consistency_dataset = file.createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, run_dataspace);
        consistency_dataset.write(consistency_values.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Save overall statistics as attributes
        H5::DataSpace attr_dataspace(H5S_SCALAR);
        H5::Attribute overall_mean_attr = file.createAttribute("overall_mean", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute overall_variance_attr = file.createAttribute("overall_variance", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute theoretical_mean_attr = file.createAttribute("theoretical_mean", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute theoretical_variance_attr = file.createAttribute("theoretical_variance", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute V0_attr = file.createAttribute("V0", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute meas_noise_std_attr = file.createAttribute("meas_noise_std", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute use_existing_data_attr = file.createAttribute("use_existing_data", H5::PredType::NATIVE_INT, attr_dataspace);
        H5::Attribute total_dof_attr = file.createAttribute("total_degrees_of_freedom", H5::PredType::NATIVE_INT, attr_dataspace);
        H5::Attribute method_attr = file.createAttribute("method", H5::PredType::C_S1, H5::DataSpace(H5S_SCALAR));
        
        overall_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_mean);
        overall_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_variance);
        theoretical_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_mean);
        theoretical_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_variance);
        V0_attr.write(H5::PredType::NATIVE_DOUBLE, &best_V0);
        meas_noise_std_attr.write(H5::PredType::NATIVE_DOUBLE, &best_meas_noise_std);
        int use_existing_flag = use_existing_data ? 1 : 0;
        use_existing_data_attr.write(H5::PredType::NATIVE_INT, &use_existing_flag);
        total_dof_attr.write(H5::PredType::NATIVE_INT, &total_dof);
        std::string method_str = "full_system_covariance";
        method_attr.write(H5::PredType::C_S1, method_str.c_str());
        
        file.close();
        std::cout << consistency_method << " validation results saved to " << h5_filename << std::endl;
        
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
#include "2D_factor_graph_trajectory.h"
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

// Function to read the best Q and R values from the optimization results file
bool load_best_params(double& best_q, double& best_r) {
    std::ifstream infile("../H5_Files/2D_bayesopt_best.txt");
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open 2D_bayesopt_best.txt to read Q and R values." << std::endl;
        std::cerr << "Please ensure the file exists and contains the best parameters." << std::endl;
        return false;
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string temp;
        char comma;

        // Expects format: "Best Q: 0.123, Best R: 4.56, ..."
        if (ss >> temp >> temp >> best_q >> comma >> temp >> temp >> best_r) {
             std::cout << "Successfully loaded Best Q: " << best_q << " and Best R: " << best_r << std::endl;
             return true;
        }
    }
    
    std::cerr << "Error: Could not parse Q and R from 2D_bayesopt_best.txt." << std::endl;
    return false;
}

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");

    // --- PART 1: SETUP ---

    double best_Q = config["validate_filter"]["Q"].as<double>();
    double best_R = config["validate_filter"]["R"].as<double>();
    
    // if (!load_best_params(best_Q, best_R)) {
    //     return 1;
    // }

    // Parameters for new data generation
    int N = config["Data_Generation"]["trajectory_length"].as<int>();
    int num_graphs = config["Data_Generation"]["num_graphs"].as<int>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    Eigen::Vector2d pos(config["Data_Generation"]["pos"]["x"].as<double>(), config["Data_Generation"]["pos"]["y"].as<double>());
    Eigen::Vector2d vel(config["Data_Generation"]["vel"]["x"].as<double>(), config["Data_Generation"]["vel"]["y"].as<double>());
    double process_noise_std = sqrt(config["Data_Generation"]["process_noise_std"].as<double>()); 
    double meas_noise_std = sqrt(config["Data_Generation"]["meas_noise_std"].as<double>());    
    unsigned int base_seed = 1339; // New seed for validation data
    int nx = 4;

    // --- PART 2: GENERATE NEW, INDEPENDENT DATA IN MEMORY ---
    
    std::cout << "\nGenerating " << num_graphs << " new trajectories for validation..." << std::endl;

    std::vector<std::vector<Eigen::Vector4d>> all_true_states(num_graphs, std::vector<Eigen::Vector4d>(N));
    std::vector<std::vector<Eigen::Vector2d>> all_measurements(num_graphs, std::vector<Eigen::Vector2d>(N));

    for (int run = 0; run < num_graphs; ++run) {
        all_true_states[run][0] << pos.x(), pos.y(), vel.x(), vel.y();
        for (int k = 1; k < N; ++k) {
            all_true_states[run][k].head<2>() = all_true_states[run][k-1].head<2>() + all_true_states[run][k-1].tail<2>() * dt;
            all_true_states[run][k].tail<2>() = all_true_states[run][k-1].tail<2>();
        }

        std::mt19937 gen(base_seed + run); 
        std::normal_distribution<> noise_q(0.0, process_noise_std);
        std::vector<Eigen::Vector4d> noisy_states = all_true_states[run];
        for (int k = 1; k < N; ++k) {
            Eigen::Vector4d process_noise;
            for (int i = 0; i < 4; ++i) process_noise[i] = noise_q(gen);
            noisy_states[k] += process_noise;
        }

        std::normal_distribution<> noise_r(0.0, meas_noise_std);
        for (int k = 0; k < N; ++k) {
            all_measurements[run][k][0] = noisy_states[k][0] + noise_r(gen);
            all_measurements[run][k][1] = noisy_states[k][1] + noise_r(gen);
        }
    }
    
    std::cout << "Data generation complete." << std::endl;

    // --- PART 3: RUN FILTER AND CALCULATE NEES ---

    std::cout << "\nRunning filter with Q=" << best_Q << " and R=" << best_R << " on new data..." << std::endl;

    // Initialize accumulators for NEES statistics per Monte Carlo run
    std::vector<double> nees_sum_per_run(num_graphs, 0.0);
    std::vector<double> nees_sum_sq_per_run(num_graphs, 0.0);
    std::vector<int> nees_count_per_run(num_graphs, 0);
    std::vector<std::vector<double>> all_nees_values_per_run(num_graphs);

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * best_Q;
        fg.R_ = Eigen::Matrix2d::Identity() * best_R;
        
        // This run uses the generated measurements. The true states are passed separately for error calculation.
        fg.run(all_true_states[run], &all_measurements[run], false);

        auto est_states = fg.getAllEstimates();
        Eigen::MatrixXd infoMat = fg.getFullInformationMatrix();
        
        // Debug: Check if information matrix looks reasonable
        if (run == 0) {
            std::cout << "Debug: Information matrix size: " << infoMat.rows() << "x" << infoMat.cols() << std::endl;
            std::cout << "Debug: First 4x4 block of information matrix:" << std::endl;
            std::cout << infoMat.block<4,4>(0,0) << std::endl;
            
            // Check a few more blocks to see the pattern
            std::cout << "Debug: Middle 4x4 block (k=10):" << std::endl;
            std::cout << infoMat.block<4,4>(10*4,10*4) << std::endl;
            std::cout << "Debug: Last 4x4 block (k=19):" << std::endl;
            std::cout << infoMat.block<4,4>(19*4,19*4) << std::endl;
            
            // Check process model contribution by looking at off-diagonal blocks
            std::cout << "Debug: Process model blocks (k=0,1):" << std::endl;
            std::cout << infoMat.block<4,4>(0*4,1*4) << std::endl;
            std::cout << "Debug: Process model blocks (k=10,11):" << std::endl;
            std::cout << infoMat.block<4,4>(10*4,11*4) << std::endl;
        }

        // Store all NEES values for this run
        all_nees_values_per_run[run].reserve(N);
        
        for (int k = 0; k < N; ++k) {
            Eigen::Vector4d err = all_true_states[run][k] - est_states[k];
            Eigen::Matrix4d info_block = infoMat.block<4,4>(k*4, k*4);
            
            // NEES = err^T * P^(-1) * err = err^T * H * err (where H is the information matrix)
            double nees_k = err.transpose() * info_block * err;
            
            // Debug: Track error magnitudes and info matrix traces for first run
            if (run == 0) {
                double err_magnitude = err.norm();
                double info_trace = info_block.trace();
                printf("k=%d: err_mag=%.3f, info_trace=%.3f, NEES=%.3f\n", 
                       k, err_magnitude, info_trace, nees_k);
            }
            
            // Store NEES value for this run
            all_nees_values_per_run[run].push_back(nees_k);
            
            // Accumulate statistics per Monte Carlo run
            nees_sum_per_run[run] += nees_k;
            nees_sum_sq_per_run[run] += nees_k * nees_k;
            nees_count_per_run[run]++;
        }
    }

    // --- PART 4: ANALYZE AND REPORT RESULTS ---
    
    std::cout << "\nValidation complete. Analyzing results..." << std::endl;

    // Calculate overall statistics from accumulated values
    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    int total_count = 0;
    
    for (int run = 0; run < num_graphs; ++run) {
        total_sum += nees_sum_per_run[run];
        total_sum_sq += nees_sum_sq_per_run[run];
        total_count += nees_count_per_run[run];
    }
    
    double sample_mean = total_sum / total_count;
    double sample_variance = (total_sum_sq / total_count) - (sample_mean * sample_mean);

    // --- Theoretical Values for Chi-Squared with k=4 degrees of freedom ---
    double theoretical_mean = nx;
    double theoretical_variance = 2 * nx;

    std::cout << "\n--- NEES Validation Results ---" << std::endl;
    std::cout << "Total NEES samples calculated: " << total_count << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "              |  Calculated  |  Theoretical (\u03C7\u00B2, k=4)" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    printf("Mean          | %12.4f | %12.4f\n", sample_mean, theoretical_mean);
    printf("Variance      | %12.4f | %12.4f\n", sample_variance, theoretical_variance);
    std::cout << "---------------------------------" << std::endl;
    
    // Calculate NEES means and variances per Monte Carlo run
    std::vector<double> nees_means_per_run(num_graphs);
    std::vector<double> nees_variances_per_run(num_graphs);
    
    for (int run = 0; run < num_graphs; ++run) {
        nees_means_per_run[run] = nees_sum_per_run[run] / nees_count_per_run[run];
        double mean_run = nees_means_per_run[run];
        nees_variances_per_run[run] = (nees_sum_sq_per_run[run] / nees_count_per_run[run]) - (mean_run * mean_run);
    }
    

    
    // Save NEES statistics to HDF5 file
    try {
        const std::string h5_filename = "../H5_Files/2D_nees_validation_results.h5";
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
        H5::Attribute Q_attr = file.createAttribute("Q", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        H5::Attribute R_attr = file.createAttribute("R", H5::PredType::NATIVE_DOUBLE, attr_dataspace);
        
        overall_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_mean);
        overall_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &sample_variance);
        theoretical_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_mean);
        theoretical_variance_attr.write(H5::PredType::NATIVE_DOUBLE, &theoretical_variance);
        Q_attr.write(H5::PredType::NATIVE_DOUBLE, &best_Q);
        R_attr.write(H5::PredType::NATIVE_DOUBLE, &best_R);
        
        file.close();
        std::cout << "NEES validation results saved to " << h5_filename << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "Error saving to HDF5: " << e.getDetailMsg() << std::endl;
    }

    return 0;
} 
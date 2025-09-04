#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

static void build_dt_vec_from_yaml(const YAML::Node &config, int T, std::vector<double> &dt_vec) {
    double dt = config["Data_Generation"]["dt"].as<double>();
    dt_vec.assign(std::max(0, T - 1), dt);
    if (config["Data_Generation"]["dt_pieces"]) {
        for (const auto &piece : config["Data_Generation"]["dt_pieces"]) {
            int from = piece["from"].as<int>();
            int to = piece["to"].as<int>();
            double dt_piece = piece["dt"].as<double>();
            from = std::max(0, from);
            to = std::min(T - 2, to);
            for (int k = from; k <= to; ++k) dt_vec[k] = dt_piece;
        }
    }
}

// Compute CNIS components for a given number of runs
struct CNISResult {
    double nis_mean;
    double nis_variance;
    double log_mean_component;
    double log_variance_component;
    double cnis;
    int dof;
};

CNISResult computeCNISForRuns(
    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
    const std::vector<double> &dt_vec,
    double dt,
    double q,
    double R,
    const std::string &method,
    int num_runs_to_use)
{
    CNISResult result;
    const int T = static_cast<int>(all_states[0].size());
    const double Rstd = std::sqrt(R);
    
    // Compute chi2 for the specified number of runs
    std::vector<double> chi2_values(num_runs_to_use);
    
    #pragma omp parallel for
    for (int run = 0; run < num_runs_to_use; ++run) {
        FactorGraph2DTrajectory fg;
        fg.setQFromProcessNoiseIntensity(q, dt);
        fg.setRFromMeasurementNoise(Rstd, Rstd);

        bool do_optimization = (method == "nis4");
        if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
            fg.run(all_states[run], &all_measurements[run], dt_vec, do_optimization);
        } else {
            fg.run(all_states[run], &all_measurements[run], dt, do_optimization);
        }
        chi2_values[run] = fg.getChi2();
    }
    
    // Calculate mean and variance
    double mean = 0.0;
    for (int run = 0; run < num_runs_to_use; ++run) {
        mean += chi2_values[run];
    }
    mean /= static_cast<double>(num_runs_to_use);
    
    double var = 0.0;
    for (int run = 0; run < num_runs_to_use; ++run) {
        double d = chi2_values[run] - mean;
        var += d * d;
    }
    if (num_runs_to_use > 1) {
        var /= static_cast<double>(num_runs_to_use - 1);
    }
    
    // Get DOF from a temporary run
    FactorGraph2DTrajectory tmp;
    tmp.setQFromProcessNoiseIntensity(q, dt);
    tmp.setRFromMeasurementNoise(Rstd, Rstd);
    if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
        tmp.run(all_states[0], &all_measurements[0], dt_vec, (method == "nis4"));
    } else {
        tmp.run(all_states[0], &all_measurements[0], dt, (method == "nis4"));
    }
    auto [dimZ, dimX] = tmp.getActualGraphDimensions();
    int dof = (method == "nis4") ? (dimZ - dimX) : dimZ;
    
    // Calculate CNIS components
    double log_mean = std::log(std::max(1e-12, mean / dof));
    double log_var_term = (num_runs_to_use > 1) ? (var / (2.0 * dof)) : 0.0;
    double log_variance = (var > 0.0) ? std::log(std::max(1e-12, log_var_term)) : 0.0;
    double cnis = std::abs(log_mean) + std::abs(log_variance);
    
    result.nis_mean = mean;
    result.nis_variance = var;
    result.log_mean_component = log_mean;
    result.log_variance_component = log_variance;
    result.cnis = cnis;
    result.dof = dof;
    
    return result;
}

int main() {
    std::cout << "=== CNIS Convergence Test for Linear Tracking ===" << std::endl;
    
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");

    std::string states_file = "../2D-Tracking/Saved_Data/2D_noisy_states.h5";
    std::string meas_file = "../2D-Tracking/Saved_Data/2D_noisy_measurements.h5";

    auto all_states = load_all_noisy_states_h5(states_file);
    auto all_measurements = load_all_noisy_measurements_h5(meas_file);

    int total_runs = static_cast<int>(all_states.size());
    if (total_runs < 50000) {
        std::cerr << "ERROR: Need at least 50,000 runs, but only found " << total_runs << std::endl;
        return 1;
    }
    
    int T = static_cast<int>(all_states[0].size());
    std::vector<double> dt_vec; 
    build_dt_vec_from_yaml(config, T, dt_vec);
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    // Get ground truth parameters
    double ground_truth_q = config["Data_Generation"]["q"].as<double>();
    double ground_truth_R = config["Data_Generation"]["meas_noise_var"].as<double>();
    std::string method = config["bayesopt"]["consistency_method"].as<std::string>("nis3");
    
    std::cout << "Total runs available: " << total_runs << std::endl;
    std::cout << "Trajectory length: " << T << std::endl;
    std::cout << "Ground truth q: " << ground_truth_q << std::endl;
    std::cout << "Ground truth R: " << ground_truth_R << std::endl;
    std::cout << "Method: " << method << std::endl;
    
    // Parameters for convergence test
    const int step_size = 100;
    const int max_runs = 50000;
    const int num_steps = max_runs / step_size; // 500 data points
    
    std::cout << "Testing convergence from " << step_size << " to " << max_runs 
              << " runs in steps of " << step_size << " (" << num_steps << " data points)" << std::endl;
    
    // Storage for results
    std::vector<int> num_runs_vec(num_steps);
    std::vector<double> nis_means(num_steps);
    std::vector<double> nis_variances(num_steps);
    std::vector<double> log_mean_components(num_steps);
    std::vector<double> log_variance_components(num_steps);
    std::vector<double> cnis_values(num_steps);
    std::vector<int> dof_values(num_steps);
    
    std::cout << "Computing CNIS convergence..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Compute CNIS for each step
    #pragma omp parallel for
    for (int step = 0; step < num_steps; ++step) {
        int num_runs_to_use = (step + 1) * step_size; // 100, 200, 300, ..., 50000
        
        CNISResult result = computeCNISForRuns(
            all_states, all_measurements, dt_vec, dt,
            ground_truth_q, ground_truth_R, method, num_runs_to_use
        );
        
        num_runs_vec[step] = num_runs_to_use;
        nis_means[step] = result.nis_mean;
        nis_variances[step] = result.nis_variance;
        log_mean_components[step] = result.log_mean_component;
        log_variance_components[step] = result.log_variance_component;
        cnis_values[step] = result.cnis;
        dof_values[step] = result.dof;
        
        if ((step + 1) % 50 == 0) {
            std::cout << "Completed step " << (step + 1) << "/" << num_steps 
                      << " (N=" << num_runs_to_use << ", CNIS=" << result.cnis << ")" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Computation completed in " << duration.count() << " seconds." << std::endl;
    
    // Save results to CSV for plotting
    std::string csv_filename = "../2D-Tracking/Saved_Data/cnis_convergence_linear.csv";
    std::ofstream csv(csv_filename);
    csv << "num_runs,nis_mean,nis_variance,log_mean_component,log_variance_component,cnis,dof\n";
    
    for (int step = 0; step < num_steps; ++step) {
        csv << num_runs_vec[step] << ","
            << nis_means[step] << ","
            << nis_variances[step] << ","
            << log_mean_components[step] << ","
            << log_variance_components[step] << ","
            << cnis_values[step] << ","
            << dof_values[step] << "\n";
    }
    csv.close();
    
    std::cout << "Results saved to: " << csv_filename << std::endl;
    
    // Save results to HDF5 as well
    try {
        std::string h5_filename = "../2D-Tracking/Saved_Data/cnis_convergence_linear.h5";
        H5::H5File file(h5_filename, H5F_ACC_TRUNC);
        hsize_t dims[1] = {static_cast<hsize_t>(num_steps)};
        H5::DataSpace dataspace(1, dims);
        
        H5::DataSet runs_ds = file.createDataSet("num_runs", H5::PredType::NATIVE_INT, dataspace);
        H5::DataSet mean_ds = file.createDataSet("nis_mean", H5::PredType::NATIVE_DOUBLE, dataspace);
        H5::DataSet var_ds = file.createDataSet("nis_variance", H5::PredType::NATIVE_DOUBLE, dataspace);
        H5::DataSet log_mean_ds = file.createDataSet("log_mean_component", H5::PredType::NATIVE_DOUBLE, dataspace);
        H5::DataSet log_var_ds = file.createDataSet("log_variance_component", H5::PredType::NATIVE_DOUBLE, dataspace);
        H5::DataSet cnis_ds = file.createDataSet("cnis", H5::PredType::NATIVE_DOUBLE, dataspace);
        H5::DataSet dof_ds = file.createDataSet("dof", H5::PredType::NATIVE_INT, dataspace);
        
        runs_ds.write(num_runs_vec.data(), H5::PredType::NATIVE_INT);
        mean_ds.write(nis_means.data(), H5::PredType::NATIVE_DOUBLE);
        var_ds.write(nis_variances.data(), H5::PredType::NATIVE_DOUBLE);
        log_mean_ds.write(log_mean_components.data(), H5::PredType::NATIVE_DOUBLE);
        log_var_ds.write(log_variance_components.data(), H5::PredType::NATIVE_DOUBLE);
        cnis_ds.write(cnis_values.data(), H5::PredType::NATIVE_DOUBLE);
        dof_ds.write(dof_values.data(), H5::PredType::NATIVE_INT);
        
        // Add metadata attributes
        H5::Attribute q_attr = file.createAttribute("ground_truth_q", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        H5::Attribute r_attr = file.createAttribute("ground_truth_R", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
        H5::Attribute method_attr = file.createAttribute("method", H5::StrType(H5::PredType::C_S1, method.length()), H5::DataSpace());
        q_attr.write(H5::PredType::NATIVE_DOUBLE, &ground_truth_q);
        r_attr.write(H5::PredType::NATIVE_DOUBLE, &ground_truth_R);
        method_attr.write(H5::StrType(H5::PredType::C_S1, method.length()), method.c_str());
        
        file.close();
        std::cout << "Results also saved to: " << h5_filename << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to save HDF5: " << e.what() << std::endl;
    }
    
    // Print summary statistics
    std::cout << "\n=== CONVERGENCE SUMMARY ===" << std::endl;
    std::cout << "Final values (N=" << max_runs << "):" << std::endl;
    std::cout << "  NIS mean: " << nis_means.back() << std::endl;
    std::cout << "  NIS variance: " << nis_variances.back() << std::endl;
    std::cout << "  Log mean component: " << log_mean_components.back() << std::endl;
    std::cout << "  Log variance component: " << log_variance_components.back() << std::endl;
    std::cout << "  CNIS: " << cnis_values.back() << std::endl;
    std::cout << "  DOF: " << dof_values.back() << std::endl;
    
    // Calculate relative change in last 10% of data to assess convergence
    int convergence_window = num_steps / 10; // Last 10% of steps
    double cnis_start = cnis_values[num_steps - convergence_window];
    double cnis_end = cnis_values[num_steps - 1];
    double relative_change = std::abs((cnis_end - cnis_start) / cnis_start) * 100.0;
    
    std::cout << "\nConvergence assessment:" << std::endl;
    std::cout << "  CNIS relative change in last 10% of data: " << relative_change << "%" << std::endl;
    std::cout << "  " << (relative_change < 1.0 ? "GOOD" : "POOR") << " convergence (threshold: 1%)" << std::endl;
    
    return 0;
} 
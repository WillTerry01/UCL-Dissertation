#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "H5Cpp.h"
#include <vector>

int main() {
    std::cout << "=== Demonstrating Tuned Parameters ===" << std::endl;
    
    // Load tuned parameters from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");
    
    double tuned_q = config["validate_filter"]["q"].as<double>();
    double tuned_R = config["validate_filter"]["R"].as<double>();
    double min_objective = config["validate_filter"]["min_objective"].as<double>();
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    std::cout << "Tuned Parameters:" << std::endl;
    std::cout << "  Process noise intensity (q): " << tuned_q << std::endl;
    std::cout << "  Measurement noise variance (R): " << tuned_R << std::endl;
    std::cout << "  Best CNEES objective: " << min_objective << std::endl;
    std::cout << "  Time step (dt): " << dt << std::endl;
    std::cout << std::endl;
    
    // Load a few sample trajectories from existing data
    auto all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_noisy_measurements.h5");
    
    int num_demo_runs = std::min(5, (int)all_states.size());  // Demo with first 5 runs
    int T = all_states[0].size();  // Trajectory length
    
    std::cout << "Running factor graph optimization on " << num_demo_runs << " demo trajectories..." << std::endl;
    
    // Prepare data structures for saving results
    std::vector<std::vector<Eigen::Vector4d>> demo_true_states;
    std::vector<std::vector<Eigen::Vector2d>> demo_measurements;
    std::vector<std::vector<Eigen::Vector4d>> demo_estimates;
    std::vector<double> demo_chi2_values;
    
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
        
        // Run optimization
        fg.run(all_states[run], &all_measurements[run], dt);
        
        // Get results
        auto est_states = fg.getAllEstimates();
        auto true_states = fg.getAllTrueStates();
        double chi2 = fg.getChi2();
        
        // Store results
        demo_true_states.push_back(true_states);
        demo_measurements.push_back(all_measurements[run]);
        demo_estimates.push_back(est_states);
        demo_chi2_values.push_back(chi2);
        
        // Print some statistics for this run
        double mean_error = 0.0;
        for (int k = 0; k < T; ++k) {
            Eigen::Vector4d error = true_states[k] - est_states[k];
            mean_error += error.head<2>().norm();  // Position error only
        }
        mean_error /= T;
        
        std::cout << "  Run " << (run + 1) << " - Chi2: " << chi2 
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
    
    return 0;
} 
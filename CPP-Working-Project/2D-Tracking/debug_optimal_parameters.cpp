#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <yaml-cpp/yaml.h>

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");
    
    // Get dt from data generation config
    double dt = config["Data_Generation"]["dt"].as<double>();
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>("nis4");  // Use same method as BO
    
    // Load the data that was generated with specific parameters
    auto all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_noisy_measurements.h5");
    
    std::cout << "=== DEBUGGING OPTIMAL PARAMETER PERFORMANCE ===" << std::endl;
    std::cout << "Testing different parameter combinations on the same dataset..." << std::endl;
    std::cout << "Consistency method: " << consistency_method << std::endl;
    std::cout << "Number of runs: " << all_states.size() << std::endl;
    std::cout << "Trajectory length: " << all_states[0].size() << std::endl;
    
    // Get the actual data generation parameters from YAML
    double data_gen_q = config["Data_Generation"]["q"].as<double>();
    double data_gen_R = config["Data_Generation"]["meas_noise_var"].as<double>();
    
    std::cout << "YAML Data Generation Parameters: q=" << data_gen_q << ", R=" << data_gen_R << std::endl;
    
    // CRITICAL DIAGNOSTIC: Check if data was actually generated with these parameters
    std::cout << "\n=== DIAGNOSTIC: Verifying Data Generation Parameters ===" << std::endl;
    
    // Test fine grid around YAML parameters to verify they should be optimal
    std::vector<std::pair<double, double>> diagnostic_params;
    for (double q_mult = 0.9; q_mult <= 1.1; q_mult += 0.05) {
        for (double R_mult = 0.9; R_mult <= 1.1; R_mult += 0.05) {
            diagnostic_params.push_back({data_gen_q * q_mult, data_gen_R * R_mult});
        }
    }
    
    std::cout << "Testing " << diagnostic_params.size() << " parameter combinations around YAML values..." << std::endl;
    
    double diagnostic_best_score = 1e6;
    std::pair<double, double> diagnostic_best_params;
    
    for (const auto& params : diagnostic_params) {
        try {
            FactorGraph2DTrajectory fg;
            fg.setQFromProcessNoiseIntensity(params.first, dt);
            fg.setRFromMeasurementNoise(sqrt(params.second), sqrt(params.second));
            
            bool do_optimization = (consistency_method == "nis4");
            fg.run(all_states[0], &all_measurements[0], dt, do_optimization);
            
            double chi2 = fg.getChi2();
            auto [dimZ, dimX] = fg.getActualGraphDimensions();
            int dof = (consistency_method == "nis3") ? dimZ : (dimZ - dimX);
            double consistency_metric = std::abs(std::log(chi2 / dof));
            
            if (consistency_metric < diagnostic_best_score) {
                diagnostic_best_score = consistency_metric;
                diagnostic_best_params = params;
            }
        } catch (...) {
            // Skip failed evaluations
        }
    }
    
    std::cout << "YAML params: q=" << data_gen_q << ", R=" << data_gen_R << std::endl;
    std::cout << "Best found: q=" << diagnostic_best_params.first << ", R=" << diagnostic_best_params.second 
              << " (score=" << diagnostic_best_score << ")" << std::endl;
    
    if (std::abs(diagnostic_best_params.first - data_gen_q) < 0.01 && 
        std::abs(diagnostic_best_params.second - data_gen_R) < 0.01) {
        std::cout << "✅ YAML parameters are locally optimal" << std::endl;
    } else {
        std::cout << "❌ YAML parameters are NOT locally optimal - DATA MISMATCH!" << std::endl;
    }
    
    // Test different parameter combinations
    std::vector<std::pair<double, double>> test_params = {
        {data_gen_q, data_gen_R},  // YAML data generation parameters (should be best)
        {0.1, 2.0},                // Lower process noise
        {1.0, 2.0},                // Higher process noise  
        {0.5, 1.0},                // Lower measurement noise
        {0.5, 4.0},                // Higher measurement noise
        {0.509253, 1.95274}        // NEW BO found parameters
    };
    
    std::vector<double> consistency_results;
    
    for (const auto& params : test_params) {
        double q_test = params.first;
        double r_test = params.second;
        
        std::cout << "\n--- Testing q=" << q_test << ", R=" << r_test << " ---" << std::endl;
        
        // Use ALL runs to match BO script exactly - calculate chi2 for all runs to get mean and variance
        int num_test_runs = all_states.size();
        std::vector<double> chi2_values;
        
        for (int run = 0; run < num_test_runs; ++run) {
            try {
                FactorGraph2DTrajectory fg;
                fg.setQFromProcessNoiseIntensity(q_test, dt);
                fg.setRFromMeasurementNoise(sqrt(r_test), sqrt(r_test));
                
                bool do_optimization = (consistency_method == "nis4");
                fg.run(all_states[run], &all_measurements[run], dt, do_optimization);
                
                double chi2 = fg.getChi2();
                chi2_values.push_back(chi2);
                
                if (run < 3) {  // Print details for first few runs
                    std::cout << "  Run " << run << ": chi2 = " << chi2 << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "  Run " << run << " failed: " << e.what() << std::endl;
            }
        }
        
        if (chi2_values.size() > 0) {
            // Calculate mean chi2
            double mean_chi2 = 0.0;
            for (double chi2 : chi2_values) {
                mean_chi2 += chi2;
            }
            mean_chi2 /= chi2_values.size();
            
            // Calculate variance of chi2
            double variance_chi2 = 0.0;
            for (double chi2 : chi2_values) {
                double diff = chi2 - mean_chi2;
                variance_chi2 += diff * diff;
            }
            if (chi2_values.size() > 1) {
                variance_chi2 /= (chi2_values.size() - 1);
            }
            
            // Get DOF for normalization
            FactorGraph2DTrajectory temp_fg;
            temp_fg.setQFromProcessNoiseIntensity(q_test, dt);
            temp_fg.setRFromMeasurementNoise(sqrt(r_test), sqrt(r_test));
            bool do_optimization_temp = (consistency_method == "nis4");
            temp_fg.run(all_states[0], &all_measurements[0], dt, do_optimization_temp);
            auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();
            
            int total_dof = (consistency_method == "nis3") ? dimZ : (dimZ - dimX);
            double normalized_chi2 = mean_chi2 / total_dof;
            
            std::cout << "  Mean chi2: " << mean_chi2 << std::endl;
            std::cout << "  Variance chi2: " << variance_chi2 << std::endl;
            std::cout << "  DOF: " << total_dof << std::endl;
            std::cout << "  Normalized chi2: " << normalized_chi2 << " (should be ~1.0 for optimal)" << std::endl;
            std::cout << "  Successful runs: " << chi2_values.size() << "/" << num_test_runs << std::endl;
            
            // Calculate CNIS exactly like BO script
            double log_mean = std::log(mean_chi2 / total_dof);
            double log_variance = (variance_chi2 > 0) ? std::log(variance_chi2 / (2.0 * total_dof)) : 0.0;
            double CNIS = std::abs(log_mean) + std::abs(log_variance);
            
            consistency_results.push_back(CNIS);
            std::cout << "  CNIS metric: " << CNIS << " (lower is better)" << std::endl;
            
        } else {
            std::cout << "  ALL RUNS FAILED!" << std::endl;
            consistency_results.push_back(1e6);  // Very bad score
        }
    }
    
    // Find best parameters
    std::cout << "\n=== SUMMARY ===" << std::endl;
    double best_score = *std::min_element(consistency_results.begin(), consistency_results.end());
    int best_idx = std::distance(consistency_results.begin(), 
                                std::min_element(consistency_results.begin(), consistency_results.end()));
    
    std::cout << "Parameter test results (consistency metric - lower is better):" << std::endl;
    for (size_t i = 0; i < test_params.size(); ++i) {
        std::string label = (i == 0) ? " <- DATA GENERATION PARAMS" : 
                           (i == test_params.size()-1) ? " <- BO FOUND PARAMS" : "";
        std::string best_marker = (i == best_idx) ? " *** BEST ***" : "";
        
        std::cout << "  q=" << test_params[i].first << ", R=" << test_params[i].second 
                  << ": " << consistency_results[i] << label << best_marker << std::endl;
    }
    
    if (best_idx == 0) {
        std::cout << "\n✅ GOOD: Data generation parameters are optimal (as expected)" << std::endl;
    } else {
        std::cout << "\n❌ PROBLEM: Data generation parameters are NOT optimal!" << std::endl;
        std::cout << "This suggests a fundamental issue with the approach." << std::endl;
    }
    
    // Additional analysis
    std::cout << "\n=== DETAILED ANALYSIS ===" << std::endl;
    std::cout << "Chi2/DOF analysis:" << std::endl;
    std::cout << "• Values close to 1.0 indicate good parameter fit" << std::endl;
    std::cout << "• Values >> 1.0 suggest underestimated noise" << std::endl;
    std::cout << "• Values << 1.0 suggest overestimated noise" << std::endl;
    std::cout << "• Small differences (< 5%) are normal optimization improvements" << std::endl;
    
    return 0;
} 
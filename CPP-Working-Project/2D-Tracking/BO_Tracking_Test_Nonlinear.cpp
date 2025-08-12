#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>
#include "H5Cpp.h"
#include <array>
#include <yaml-cpp/yaml.h>

// Define Hyper Parameters for BAYESOPT
void initialisation(bopt_params &params, const YAML::Node& config) {
    params.n_iterations = config["bayesopt"]["n_iterations"].as<int>();
    params.n_init_samples = config["bayesopt"]["n_init_samples"].as<int>();
    params.crit_name = strdup(config["bayesopt"]["crit_name"].as<std::string>().c_str());
    params.verbose_level = config["bayesopt"]["verbose_level"].as<int>();
    params.surr_name = strdup(config["bayesopt"]["surr_name"].as<std::string>().c_str());
    params.noise = config["bayesopt"]["noise"].as<double>();
    
    // Enhanced exploration parameters
    if (config["bayesopt"]["force_jump"]) {
        params.force_jump = config["bayesopt"]["force_jump"].as<int>();
    }
    if (config["bayesopt"]["n_inner_iterations"]) {
        params.n_inner_iterations = config["bayesopt"]["n_inner_iterations"].as<int>();
    }
    // For Expected Improvement kappa parameter
    if (config["bayesopt"]["ei_kappa"]) {
        params.crit_params[0] = config["bayesopt"]["ei_kappa"].as<double>();
    }
    // For LCB alpha parameter (if switching back)
    if (config["bayesopt"]["lcb_alpha"]) {
        params.crit_params[0] = config["bayesopt"]["lcb_alpha"].as<double>();
    }
}

// Helper function to infer num_graphs and N from the HDF5 file
void infer_problem_size_h5(const std::string& filename, int& num_graphs, int& Trajectory_length) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("states");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims, nullptr);
        num_graphs = dims[0];
        Trajectory_length = dims[1];
    } catch (H5::Exception& e) {
        std::cerr << "Error inferring problem size from HDF5: " << e.getDetailMsg() << std::endl;
        num_graphs = 0;
        Trajectory_length = 0;
    }
}

class CMetricBayesOptNonlinear : public bayesopt::ContinuousModel {
public:
    CMetricBayesOptNonlinear(const bopt_params &params,
                             const std::vector<std::vector<Eigen::Vector4d>> &all_states,
                             const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
                             std::vector<std::array<double, 3>> &trials,
                             double lower_bound_Q, double upper_bound_Q,
                             double lower_bound_R, double upper_bound_R,
                             double dt,
                             const std::string &consistency_method,
                             double turn_rate)
        : bayesopt::ContinuousModel(2, params),
          all_states_(all_states),
          all_measurements_(all_measurements),
          trials_(trials),
          lower_bound_Q_(lower_bound_Q),
          upper_bound_Q_(upper_bound_Q),
          lower_bound_R_(lower_bound_R),
          upper_bound_R_(upper_bound_R),
          dt_(dt),
          consistency_method_(consistency_method),
          turn_rate_(turn_rate) {}

    double evaluateSample(const boost::numeric::ublas::vector<double> &query) override {
        static int eval_count = 0;
        eval_count++;
        std::cout << "[BayesOpt Nonlinear] Iteration: " << eval_count << std::endl;
        auto eval_start = std::chrono::high_resolution_clock::now();
        
        double V0 = query(0);  // Process noise intensity for x-direction
        double V1 = query(0);  // Process noise intensity for y-direction (same as V0 for now)
        double meas_noise_var = query(1);  // Measurement noise variance (R)
        double meas_noise_std = sqrt(meas_noise_var);  // Measurement noise standard deviation
        double consistency_metric = 1e6;
        
        if (V0 >= lower_bound_Q_ && V0 <= upper_bound_Q_ && meas_noise_var >= lower_bound_R_ && meas_noise_var <= upper_bound_R_) {
            int num_graphs = all_states_.size();
            int T = all_states_[0].size();
            int nx = 4; // state dimension
            int nz = 2; // measurement dimension
            
            // Calculate theoretical DOF values for comparison
            if (eval_count <= 3) {
                std::cout << "Theoretical DOF: T*nx = " << T * nx << ", T*nz = " << T * nz << std::endl;
            }
            
            if (consistency_method_ == "cnees") {
                consistency_metric = computeCNEES(num_graphs, T, nx, nz, V0, meas_noise_std, eval_count);
            } else if (consistency_method_ == "nis3" || consistency_method_ == "nis4") {
                consistency_metric = computeCNIS(num_graphs, T, nx, nz, V0, meas_noise_std, eval_count);
            } else {
                std::cerr << "Unknown consistency method: " << consistency_method_ << std::endl;
                return 1e6;
            }
        }
        
        // Store trial results
        std::array<double, 3> trial_result = {V0, meas_noise_var, consistency_metric};
        trials_.push_back(trial_result);
        
        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_duration = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);
        
        std::cout << "[BayesOpt Nonlinear] Parameters: V0=" << V0 << ", R=" << meas_noise_var 
                  << ", Metric=" << consistency_metric << ", Time=" << eval_duration.count() << "ms" << std::endl;
        
        return consistency_metric;
    }

    // Method to compute CNEES (Normalized Estimation Error Squared)
    double computeCNEES(int num_graphs, int T, int nx, int nz, double V0, double meas_noise_std, int eval_count) {
        if (eval_count <= 1) {
            std::cout << "Computing CNEES using NEES (state estimation error) for NONLINEAR system..." << std::endl;
        }
        
        // Step 1: Compute NEES for each run using full system covariance
        std::vector<double> nees_full_system(num_graphs, 0.0);
        #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            // Create thread-local factor graph to avoid race conditions
            FactorGraph2DTrajectory fg;
            
            // Set up NONLINEAR system
            fg.setMotionModelType("constant_turn_rate", turn_rate_);
            fg.setMeasurementModelType("gps");  // GPS Cartesian position measurements
            
            fg.setQFromProcessNoiseIntensity(V0, dt_);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
                
            FactorGraph2DTrajectory::OutputOptions opts;
            opts.output_estimated_state = true;
            opts.output_true_state = true;
            opts.output_information_matrix = true;
            fg.setOutputOptions(opts);
            
            // Use NONLINEAR optimization
            fg.runNonlinear(all_states_[run], &all_measurements_[run], dt_);
            
            auto est_states = fg.getAllEstimates();
            auto true_states = fg.getAllTrueStates();
            Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
            
            // Check if information matrix is valid
            if (Hessian.rows() == 0 || Hessian.cols() == 0) {
                std::cerr << "ERROR: Information matrix is empty for run " << run << std::endl;
                continue;
            }
            
            // Create full error vector by stacking all time step errors
            Eigen::VectorXd full_error(T * 4);
            for (int k = 0; k < T; ++k) {
                Eigen::Vector4d err = true_states[k] - est_states[k];
                full_error.segment<4>(k * 4) = err;
            }
            
            // Calculate NEES using full system: err^T * P^(-1) * err = err^T * H * err
            double nees_full = full_error.transpose() * Hessian * full_error;
            nees_full_system[run] = nees_full;
        }
        
        // Step 2: Calculate mean and variance across all runs
        double mean_nees = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            mean_nees += nees_full_system[run];
        }
        mean_nees /= num_graphs;
        
        double variance_nees = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            double diff = nees_full_system[run] - mean_nees;
            variance_nees += diff * diff;
        }
        if (num_graphs > 1) {
            variance_nees /= (num_graphs - 1);
        }

        // Step 3: DOF calculation for NEES using actual graph dimensions
        FactorGraph2DTrajectory temp_fg;
        temp_fg.setMotionModelType("constant_turn_rate", turn_rate_);
        temp_fg.setMeasurementModelType("gps");  // GPS Cartesian position measurements
        temp_fg.setQFromProcessNoiseIntensity(V0, dt_);
        temp_fg.setRFromMeasurementNoise(meas_noise_std, 0.1);
        temp_fg.runNonlinear(all_states_[0], &all_measurements_[0], dt_);
        
        auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();
        int total_dof = dimX;  // For NEES, DOF = total vertex dimensions
        
        if (eval_count <= 3) {
            std::cout << "NEES DOF calculation: actual dimX = " << dimX << " (vs theoretical T*nx = " << T * nx << ")" << std::endl;
        }
        
        // Step 4: Compute CNEES
        double log_mean = std::log(mean_nees / total_dof);
        double log_variance = (variance_nees > 0) ? std::log(variance_nees / (2.0 * total_dof)) : 0.0;
        double CNEES = std::abs(log_mean) + std::abs(log_variance);
        
        return CNEES;
    }

    // Method to compute CNIS (Normalized Innovation Squared) 
    double computeCNIS(int num_graphs, int T, int nx, int nz, double V0, double meas_noise_std, int eval_count) {
        if (eval_count <= 1) {
            std::cout << "Computing CNIS using NIS (measurement innovation) for NONLINEAR system with method: " << consistency_method_ << std::endl;
        }
        
        // For nonlinear range-bearing measurements, we need to compute residuals manually
        std::vector<double> nis_values(num_graphs, 0.0);
        
        #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            FactorGraph2DTrajectory fg;
            
            // Set up NONLINEAR system
            fg.setMotionModelType("constant_turn_rate", turn_rate_);
            fg.setMeasurementModelType("gps");  // GPS Cartesian position measurements
            
            fg.setQFromProcessNoiseIntensity(V0, dt_);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
            
            // Run optimization
            bool do_optimization = (consistency_method_ == "nis4");
            fg.runNonlinear(all_states_[run], &all_measurements_[run], dt_, do_optimization);
            
            // Get chi-squared value (this represents the NIS for the entire trajectory)
            double chi2 = fg.getChi2();
            nis_values[run] = chi2;
        }
        
        // Calculate mean and variance of NIS values
        double mean_nis = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            mean_nis += nis_values[run];
        }
        mean_nis /= num_graphs;
        
        double variance_nis = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            double diff = nis_values[run] - mean_nis;
            variance_nis += diff * diff;
        }
        if (num_graphs > 1) {
            variance_nis /= (num_graphs - 1);
        }
        
        // Get actual graph dimensions for DOF calculation
        FactorGraph2DTrajectory temp_fg;
        temp_fg.setMotionModelType("constant_turn_rate", turn_rate_);
        temp_fg.setMeasurementModelType("gps");  // GPS Cartesian position measurements
        temp_fg.setQFromProcessNoiseIntensity(V0, dt_);
        temp_fg.setRFromMeasurementNoise(meas_noise_std, 0.1);
        temp_fg.runNonlinear(all_states_[0], &all_measurements_[0], dt_, false);
        
        auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();
        int total_dof;
        
        if (consistency_method_ == "nis3") {
            // Proposition 3 (MATLAB student approach): DOF = dimZ (total residuals from actual graph)
            total_dof = dimZ;
            if (eval_count <= 3) {
                std::cout << "Using Khosoussi Proposition 3 (MATLAB student approach): DOF = dimZ = " << total_dof << std::endl;
                std::cout << "  Ground truth init + no optimization + chi2 test" << std::endl;
                std::cout << "  (vs theoretical Nz = " << T * nz + (T - 1) * nx << ")" << std::endl;
            }
        } else if (consistency_method_ == "nis4") {
            // Proposition 4 (MATLAB student approach): DOF = dimZ - dimX (total residuals - total state parameters from actual graph)
            total_dof = dimZ - dimX;
            if (eval_count <= 3) {
                std::cout << "Using Khosoussi Proposition 4 (MATLAB student approach): DOF = dimZ - dimX = " << dimZ << " - " << dimX << " = " << total_dof << std::endl;
                std::cout << "  Noisy init + optimization + chi2 test" << std::endl;
                std::cout << "  (vs theoretical Nz - Nx = " << (T * nz + (T - 1) * nx) - (T * nx) << ")" << std::endl;
            }
        } else {
            // Fallback to simple nz per timestep
            total_dof = nz;
            if (eval_count <= 3) {
                std::cout << "Using simple NIS: DOF = nz = " << total_dof << std::endl;
            }
        }
        
        // Compute CNIS
        double log_mean = std::log(mean_nis / total_dof);
        double log_variance = (variance_nis > 0) ? std::log(variance_nis / (2.0 * total_dof)) : 0.0;
        double CNIS = std::abs(log_mean) + std::abs(log_variance);
        
        return CNIS;
    }

private:
    const std::vector<std::vector<Eigen::Vector4d>> &all_states_;
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements_;
    std::vector<std::array<double, 3>> &trials_;
    double lower_bound_Q_, upper_bound_Q_, lower_bound_R_, upper_bound_R_;
    double dt_;
    std::string consistency_method_;
    double turn_rate_;
};

int main() {
    std::cout << "=== Bayesian Optimization for Nonlinear Factor Graph Tracking ===" << std::endl;
    
    // Load configuration
    YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");
    
    // Get nonlinear parameters
    double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);  // rad/s
    // Removed sensor_pos - not needed for GPS tracking
    
    // Load nonlinear data
    std::string states_file = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
    std::string measurements_file = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";
    
    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    
    try {
        all_states = load_all_noisy_states_h5(states_file);
        all_measurements = load_all_noisy_measurements_h5(measurements_file);
        std::cout << "Loaded " << all_states.size() << " nonlinear trajectories" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading nonlinear data: " << e.what() << std::endl;
        std::cerr << "Please run tracking_gen_data_nonlinear first to generate nonlinear data." << std::endl;
        return -1;
    }
    
    // Get problem dimensions
    int num_graphs = all_states.size();
    int T = all_states[0].size();
    double dt = config["Data_Generation"]["dt"].as<double>();
    
    // Get parameter bounds
    double lower_bound_Q = config["parameters"][0]["lower_bound"].as<double>();
    double upper_bound_Q = config["parameters"][0]["upper_bound"].as<double>();
    double lower_bound_R = config["parameters"][1]["lower_bound"].as<double>();
    double upper_bound_R = config["parameters"][1]["upper_bound"].as<double>();
    
    // Get consistency method
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>();
    
    // Now output all the information after variables are declared
    std::cout << "=== Nonlinear BayesOpt Tracking Test ===" << std::endl;
    std::cout << "  Turn rate: " << turn_rate << " rad/s" << std::endl;
    std::cout << "  Consistency method: " << consistency_method << std::endl;
    std::cout << "  Process noise bounds: [" << lower_bound_Q << ", " << upper_bound_Q << "]" << std::endl;
    std::cout << "  Measurement noise bounds: [" << lower_bound_R << ", " << upper_bound_R << "]" << std::endl;
    std::cout << "  Time step: " << dt << " s" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    std::cout << "Problem dimensions: " << num_graphs << " trajectories, " << T << " time steps" << std::endl;
    std::cout << "Parameter bounds: Q=[" << lower_bound_Q << ", " << upper_bound_Q 
              << "], R=[" << lower_bound_R << ", " << upper_bound_R << "]" << std::endl;
    std::cout << "Consistency method: " << consistency_method << std::endl;
    
    // Set up BayesOpt parameters
    bopt_params params = initialize_parameters_to_default();
    initialisation(params, config);
    
    // Store trial results
    std::vector<std::array<double, 3>> trials;
    
    // Bounds for V0 (process noise intensity) and meas_noise_std² (measurement noise variance)
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = lower_bound_Q;
    ub(0) = upper_bound_Q;
    lb(1) = lower_bound_R;
    ub(1) = upper_bound_R;
    
    // Create Bayesian optimization object
    CMetricBayesOptNonlinear bayes_opt(params, all_states, all_measurements, trials,
                                       lower_bound_Q, upper_bound_Q, lower_bound_R, upper_bound_R,
                                       dt, consistency_method, turn_rate);
    bayes_opt.setBoundingBox(lb, ub);
    
    std::cout << "\nStarting Bayesian Optimization for Nonlinear System..." << std::endl;
    std::cout << "Number of iterations: " << params.n_iterations << std::endl;
    std::cout << "Number of initial samples: " << params.n_init_samples << std::endl;
    
    // Run optimization
    boost::numeric::ublas::vector<double> result(2);
    auto start_time = std::chrono::high_resolution_clock::now();
    bayes_opt.optimize(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nOptimization completed in " << duration.count() << " ms" << std::endl;
    
    // Get best result
    double best_objective = bayes_opt.getValueAtMinimum();
    
    std::cout << "\n=== OPTIMIZATION RESULTS ===" << std::endl;
    std::cout << "Best process noise intensity (q): " << result(0) << std::endl;
    std::cout << "Best measurement noise variance (R): " << result(1) << std::endl;
    std::cout << "Best objective value: " << best_objective << std::endl;
    
    // Save results
    std::string h5_filename = "../2D-Tracking/Saved_Data/2D_bayesopt_nonlinear_trials.h5";
    H5::H5File file(h5_filename, H5F_ACC_TRUNC);
    
    // Save trials data
    std::vector<double> q_values, r_values, objective_values;
    for (const auto& trial : trials) {
        q_values.push_back(trial[0]);
        r_values.push_back(trial[1]);
        objective_values.push_back(trial[2]);
    }
    
    hsize_t trials_dims[1] = {static_cast<hsize_t>(trials.size())};
    H5::DataSpace trials_space(1, trials_dims);
    
    H5::DataSet q_dataset = file.createDataSet("q_values", H5::PredType::NATIVE_DOUBLE, trials_space);
    H5::DataSet r_dataset = file.createDataSet("r_values", H5::PredType::NATIVE_DOUBLE, trials_space);
    H5::DataSet obj_dataset = file.createDataSet("objective_values", H5::PredType::NATIVE_DOUBLE, trials_space);
    
    q_dataset.write(q_values.data(), H5::PredType::NATIVE_DOUBLE);
    r_dataset.write(r_values.data(), H5::PredType::NATIVE_DOUBLE);
    obj_dataset.write(objective_values.data(), H5::PredType::NATIVE_DOUBLE);
    
    file.close();
    
    std::cout << "Results saved to: " << h5_filename << std::endl;
    
    // Save best parameters back to scenario_nonlinear.yaml as validation results
    std::string yaml_filename = "../scenario_nonlinear.yaml";
    
    // Read the existing YAML file
    YAML::Node yaml_config = YAML::LoadFile(yaml_filename);
    
    // Update the validation results section
    yaml_config["validate_filter"]["q"] = result(0);
    yaml_config["validate_filter"]["R"] = result(1);
    yaml_config["validate_filter"]["min_objective"] = best_objective;
    
    // Add optimization metadata
    yaml_config["validate_filter"]["optimization_date"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    yaml_config["validate_filter"]["consistency_method"] = consistency_method;
    yaml_config["validate_filter"]["turn_rate"] = turn_rate;
    
    // Write back to the YAML file
    std::ofstream yaml_file(yaml_filename);
    yaml_file << yaml_config;
    yaml_file.close();
    
    std::cout << "Best parameters saved to: " << yaml_filename << std::endl;
    std::cout << "Updated validation results:" << std::endl;
    std::cout << "  q: " << result(0) << std::endl;
    std::cout << "  R: " << result(1) << std::endl;
    std::cout << "  min_objective: " << best_objective << std::endl;
    
    return 0;
} 
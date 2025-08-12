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



class CMetricBayesOpt : public bayesopt::ContinuousModel {
public:
    CMetricBayesOpt(const bopt_params &params,
                    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
                    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
                    std::vector<std::array<double, 3>> &trials,
                    double lower_bound_Q, double upper_bound_Q,
                    double lower_bound_R, double upper_bound_R,
                    double dt,
                    const std::string &consistency_method)
        : bayesopt::ContinuousModel(2, params),
          all_states_(all_states),
          all_measurements_(all_measurements),
          trials_(trials),
          lower_bound_Q_(lower_bound_Q),
          upper_bound_Q_(upper_bound_Q),
          lower_bound_R_(lower_bound_R),
          upper_bound_R_(upper_bound_R),
          dt_(dt),
          consistency_method_(consistency_method) {}

    double evaluateSample(const boost::numeric::ublas::vector<double> &query) override {
        static int eval_count = 0;
        eval_count++;
        std::cout << "[BayesOpt] Iteration: " << eval_count << std::endl;
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
                // Calculate theoretical DOF values
                int dof_nees_theoretical = T * nx;
                int Nz_theoretical = T * nz + (T - 1) * nx;
                int Nx_theoretical = T * nx;
                int dof_nis3_theoretical = Nz_theoretical;
                int dof_nis4_theoretical = Nz_theoretical - Nx_theoretical;
                
                // Get actual graph dimensions
                FactorGraph2DTrajectory temp_fg;
                temp_fg.setQFromProcessNoiseIntensity(V0, dt_);
                temp_fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
                temp_fg.run(all_states_[0], &all_measurements_[0], dt_);
                auto [dimZ_actual, dimX_actual] = temp_fg.getActualGraphDimensions();
                
                int dof_nees_actual = dimX_actual;
                int dof_nis3_actual = dimZ_actual;
                int dof_nis4_actual = dimZ_actual - dimX_actual;
                
                std::cout << "DOF Comparison for T=" << T << ":" << std::endl;
                std::cout << "  Theoretical - NEES=" << dof_nees_theoretical 
                          << ", NIS3=" << dof_nis3_theoretical << ", NIS4=" << dof_nis4_theoretical << std::endl;
                std::cout << "  Actual Graph - NEES=" << dof_nees_actual 
                          << ", NIS3=" << dof_nis3_actual << ", NIS4=" << dof_nis4_actual << std::endl;
                std::cout << "  Using method: " << consistency_method_ << std::endl;
            }
            
            if (consistency_method_ == "nis3" || consistency_method_ == "nis4") {
                // Compute CNIS (Normalized Innovation Squared) with Khosoussi propositions
                consistency_metric = computeCNIS(num_graphs, T, nx, nz, V0, meas_noise_std, eval_count);
            } else {
                // Compute CNEES (Normalized Estimation Error Squared)  
                consistency_metric = computeCNEES(num_graphs, T, nx, nz, V0, meas_noise_std, eval_count);
            }
            
            trials_.push_back({V0, meas_noise_var, consistency_metric});
            std::cout << "V0: " << V0 << ", meas_noise_var: " << meas_noise_var 
                      << ", " << consistency_method_ << ": " << consistency_metric << std::endl;
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
            return consistency_metric;
        }
        trials_.push_back({V0, meas_noise_var, consistency_metric});
        std::cout << "V0: " << V0 << ", meas_noise_var: " << meas_noise_var 
                  << ", " << consistency_method_ << ": " << consistency_metric 
                  << " (out of bounds)" << std::endl;
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
        return consistency_metric;
    }

    // Method to compute CNEES (Normalized Estimation Error Squared)
    // NEES Formula: (x_true - x_est)^T * P_est^(-1) * (x_true - x_est)
    // Tests how well estimated states match true states
    double computeCNEES(int num_graphs, int T, int nx, int nz, double V0, double meas_noise_std, int eval_count) {
        if (eval_count <= 1) {
            std::cout << "Computing CNEES using NEES (state estimation error)..." << std::endl;
        }
        
        // Step 1: Compute NEES for each run using full system covariance
        std::vector<double> nees_full_system(num_graphs, 0.0);
        #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            // Create thread-local factor graph to avoid race conditions
            FactorGraph2DTrajectory fg;
            
            fg.setQFromProcessNoiseIntensity(V0, dt_);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
                
            FactorGraph2DTrajectory::OutputOptions opts;
            opts.output_estimated_state = true;
            opts.output_true_state = true;
            opts.output_information_matrix = true;
            fg.setOutputOptions(opts);
            fg.run(all_states_[run], &all_measurements_[run], dt_);
            auto est_states = fg.getAllEstimates();
            auto true_states = fg.getAllTrueStates();
            Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
            Eigen::MatrixXd full_cov = Hessian.inverse();
            
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

        // Step 3: DOF calculation for NEES using actual graph dimensions (MATLAB student approach)
        // Get actual graph dimensions from the first run
        FactorGraph2DTrajectory temp_fg;
        temp_fg.setQFromProcessNoiseIntensity(V0, dt_);
        temp_fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
        temp_fg.run(all_states_[0], &all_measurements_[0], dt_);
        
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
    // NIS Formula: (z_actual - z_predicted)^T * S^(-1) * (z_actual - z_predicted)
    // Where z_predicted = h(x_est), S = H*P*H^T + R
    // Tests how well predicted measurements match actual measurements
    double computeCNIS(int num_graphs, int T, int nx, int nz, double V0, double meas_noise_std, int eval_count) {
        if (eval_count <= 1) {
            std::cout << "Computing CNIS using NIS (measurement innovation) with method: " << consistency_method_ << std::endl;
        }
        
        // Measurement Jacobian H for 2D position measurements: H = [I_2x2, 0_2x2]
        Eigen::Matrix<double, 2, 4> H;
        H << 1, 0, 0, 0,    // x_pos = state[0]
             0, 1, 0, 0;    // y_pos = state[1]
        
        // Measurement noise covariance R (2x2)
        Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (meas_noise_std * meas_noise_std);
        
        // Step 1: Compute full-trajectory NIS for each run (similar to full-trajectory NEES)
        std::vector<double> nis_full_system(num_graphs, 0.0);
        
                #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            FactorGraph2DTrajectory fg;
            fg.setQFromProcessNoiseIntensity(V0, dt_);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
            
            bool do_optimization = (consistency_method_ != "nis3");
            fg.run(all_states_[run], &all_measurements_[run], dt_, do_optimization);
            nis_full_system[run] = fg.getChi2();
        }
        
        // Step 2: Calculate mean and variance across all full-trajectory NIS values  
        double mean_nis = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            mean_nis += nis_full_system[run];
        }
        mean_nis /= num_graphs;
        
        double variance_nis = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            double diff = nis_full_system[run] - mean_nis;
            variance_nis += diff * diff;
        }
        if (num_graphs > 1) {
            variance_nis /= (num_graphs - 1);
        }

        // Step 3: DOF calculation for NIS using actual graph dimensions (MATLAB student approach)
        // Get actual graph dimensions from the first run
        FactorGraph2DTrajectory temp_fg;
        temp_fg.setQFromProcessNoiseIntensity(V0, dt_);
        temp_fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
        temp_fg.run(all_states_[0], &all_measurements_[0], dt_);
        
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
        
        // Step 4: Compute CNIS 
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
};

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");

    // Get dt from data generation config
    double dt = config["Data_Generation"]["dt"].as<double>();
    std::cout << "Using dt = " << dt << " from data generation config" << std::endl;
    
    // Get consistency method from YAML config
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>("nees");
    std::cout << "Consistency method: " << consistency_method << std::endl;

    // Infer problem size from the noisy states HDF5
    int Trajectory_length = 0;
    int num_graphs = 0;
    infer_problem_size_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from HDF5." << std::endl;
        return 1;
    }
    std::cout << "Detected " << num_graphs << " runs, each of length " << Trajectory_length << std::endl;

    // Load all runs' data from HDF5
    auto all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_noisy_measurements.h5");

    // Prepare to collect trials
    std::vector<std::array<double, 3>> trials;

    // Set up BayesOpt parameters
    bopt_params params = initialize_parameters_to_default();
    initialisation(params, config);

    // Bounds for V0 (process noise intensity) and meas_noise_std² (measurement noise variance)
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = config["parameters"][0]["lower_bound"].as<double>();  // V0 lower bound
    ub(0) = config["parameters"][0]["upper_bound"].as<double>();  // V0 upper bound
    lb(1) = config["parameters"][1]["lower_bound"].as<double>();  // meas_noise_std² lower bound
    ub(1) = config["parameters"][1]["upper_bound"].as<double>();  // meas_noise_std² upper bound

    CMetricBayesOpt opt(
        params, all_states, all_measurements, trials,
        lb(0), ub(0), lb(1), ub(1), dt, consistency_method
    );
    opt.setBoundingBox(lb, ub);
    boost::numeric::ublas::vector<double> result(2);
    auto total_start = std::chrono::high_resolution_clock::now();
    opt.optimize(result);
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    double minC = opt.getValueAtMinimum();

    // Save trials to HDF5
    const std::string h5_filename = "../2D-Tracking/Saved_Data/2D_bayesopt_trials.h5";
    hsize_t dims[2] = {trials.size(), 3};
    H5::H5File file(h5_filename, H5F_ACC_TRUNC);
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = file.createDataSet("trials", H5::PredType::NATIVE_DOUBLE, dataspace);
    std::vector<double> flat_trials;
    flat_trials.reserve(trials.size() * 3);
    for (const auto& row : trials) {
        flat_trials.insert(flat_trials.end(), row.begin(), row.end());
    }
    dataset.write(flat_trials.data(), H5::PredType::NATIVE_DOUBLE);

    // Save best V0, meas_noise_std², and final objective value to BO_Parameters.yaml
    YAML::Node out_config = YAML::LoadFile("../scenario_linear.yaml");
    out_config["validate_filter"]["q"] = result(0);  // V0 (process noise intensity)
    out_config["validate_filter"]["R"] = result(1);  // meas_noise_std² (measurement noise variance)
    out_config["validate_filter"]["min_objective"] = minC;
    std::ofstream yaml_out("../scenario_linear.yaml");
    yaml_out << out_config;
    yaml_out.close();

    std::cout << "Best V0: " << result(0) << ", Best meas_noise_var: " << result(1) 
              << ", Best CNEES: " << minC << std::endl;
    std::cout << "[Timing] Total BayesOpt optimization took " << total_duration.count() << " seconds." << std::endl;
    std::ofstream cfile("../2D-Tracking/Saved_Data/2D_bayesopt_best.txt");
    cfile << "Best V0: " << result(0) << ", Best meas_noise_var: " << result(1) 
          << ", Best CNEES: " << minC << std::endl;
    cfile.close();

    return 0;
} 
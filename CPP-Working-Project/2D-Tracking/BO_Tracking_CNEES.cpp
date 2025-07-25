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
                    double dt)
        : bayesopt::ContinuousModel(2, params),
          all_states_(all_states),
          all_measurements_(all_measurements),
          trials_(trials),
          lower_bound_Q_(lower_bound_Q),
          upper_bound_Q_(upper_bound_Q),
          lower_bound_R_(lower_bound_R),
          upper_bound_R_(upper_bound_R),
          dt_(dt) {}

    double evaluateSample(const boost::numeric::ublas::vector<double> &query) override {
        static int eval_count = 0;
        eval_count++;
        std::cout << "[BayesOpt] Iteration: " << eval_count << std::endl;
        auto eval_start = std::chrono::high_resolution_clock::now();
        
        double V0 = query(0);  // Process noise intensity for x-direction
        double V1 = query(0);  // Process noise intensity for y-direction (same as V0 for now)
        double meas_noise_var = query(1);  // Measurement noise variance (R)
        double meas_noise_std = sqrt(meas_noise_var);  // Measurement noise standard deviation
        double CNEES = 1e6;
        
        if (V0 >= lower_bound_Q_ && V0 <= upper_bound_Q_ && meas_noise_var >= lower_bound_R_ && meas_noise_var <= upper_bound_R_) {
            int num_graphs = all_states_.size();
            int T = all_states_[0].size();
            int nx = 4; // state dimension
            
            // Step 1: Compute NEES for each run using full system covariance
            std::vector<double> nees_full_system(num_graphs, 0.0);
            #pragma omp parallel for
            for (int run = 0; run < num_graphs; ++run) {
                FactorGraph2DTrajectory fg;
                
                // Use proper Q matrix structure for 2D linear tracking with theoretical structure
                // V0 = std::max(V0, min_variance / dt); // Comment out this line temporarily:
                fg.setQFromProcessNoiseIntensity(V0, dt_);  // Use actual dt from data generation
                
                // Use proper R matrix structure (2x2 diagonal matrix)
                fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
                
                FactorGraph2DTrajectory::OutputOptions opts;
                opts.output_estimated_state = true;
                opts.output_true_state = true;
                opts.output_information_matrix = true;
                fg.setOutputOptions(opts);
                fg.run(all_states_[run], &all_measurements_[run], false, dt_);
                auto est_states = fg.getAllEstimates();
                auto true_states = fg.getAllTrueStates();
                Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
                Eigen::MatrixXd full_cov = Hessian.inverse();
                
                // Check if information matrix is valid
                if (Hessian.rows() == 0 || Hessian.cols() == 0) {
                    std::cerr << "ERROR: Information matrix is empty for run " << run << std::endl;
                    continue;
                }
                
                // Check if information matrix has the expected size
                if (Hessian.rows() != T * 4 || Hessian.cols() != T * 4) {
                    std::cerr << "ERROR: Information matrix has wrong size: " 
                              << Hessian.rows() << "x" << Hessian.cols() 
                              << " (expected " << T * 4 << "x" << T * 4 << ")" << std::endl;
                    continue;
                }
                
                // Create full error vector by stacking all time step errors
                Eigen::VectorXd full_error(T * 4);
                for (int k = 0; k < T; ++k) {
                    Eigen::Vector4d err = true_states[k] - est_states[k];
                    full_error.segment<4>(k * 4) = err;
                }
                
                // Check if full covariance matrix is positive definite
                Eigen::LLT<Eigen::MatrixXd> lltOfCov(full_cov);
                if (lltOfCov.info() != Eigen::Success) {
                    std::cerr << "Warning: Full covariance matrix is not positive definite for run " << run << ". Skipping NEES." << std::endl;
                    nees_full_system[run] = 0.0;
                    continue;
                }
                
                // Calculate NEES using full system: err^T * P^(-1) * err = err^T * H * err
                // Since P = full_cov and P^(-1) = Hessian, we use: err^T * Hessian * err
                double nees_full = full_error.transpose() * Hessian * full_error;
                nees_full_system[run] = nees_full;
                
                // Debug: Print detailed information for first few iterations
                if (eval_count <= 3 && run == 0) {
                    std::cout << "DEBUG: Iteration " << eval_count << ", Run " << run << std::endl;
                    std::cout << "  V0: " << V0 << ", dt: " << dt_ << std::endl;
                    std::cout << "  Process noise intensity V0: " << V0 << std::endl;
                    std::cout << "  Measurement noise std: " << meas_noise_std << std::endl;
                    std::cout << "  Full error norm: " << full_error.norm() << std::endl;
                    std::cout << "  Full covariance trace: " << full_cov.trace() << std::endl;
                    std::cout << "  Full covariance det: " << full_cov.determinant() << std::endl;
                    std::cout << "  Hessian trace: " << Hessian.trace() << std::endl;
                    std::cout << "  Hessian condition number: " << Hessian.norm() * full_cov.norm() << std::endl;
                    std::cout << "  Full NEES: " << nees_full << std::endl;
                    std::cout << "  Expected NEES (full system): " << T * nx << std::endl;
                    std::cout << "  NEES ratio (actual/expected): " << nees_full / (T * nx) << std::endl;
                    std::cout << "  ---" << std::endl;
                }
            }
            
            // Step 2: Calculate mean NEES across all runs (full system approach)
            // Mean of full system NEES values
            double mean_nees_full_system = 0.0;
            for (int run = 0; run < num_graphs; ++run) {
                mean_nees_full_system += nees_full_system[run];
            }
            mean_nees_full_system /= num_graphs;  // Average across runs
            
            // Step 3: Compute variance of full system NEES across runs
            double variance_nees_full_system = 0.0;
            for (int run = 0; run < num_graphs; ++run) {
                double diff = nees_full_system[run] - mean_nees_full_system;
                variance_nees_full_system += diff * diff;
            }
            if (num_graphs > 1) {
                variance_nees_full_system /= (num_graphs - 1);
            } else {
                variance_nees_full_system = 0.0; // Avoid division by zero if only one run
            }

            // For full system NEES: expected mean is T*nx (total degrees of freedom)
            int total_dof = T * nx;  // n_x in the formula = total degrees of freedom
            
            // Compute CNEES exactly as in the provided formula:
            // C_NEES = |log(ε̃_x / n_x)| + |log(S̃_x / (2*n_x))|
            double log_mean = std::log(mean_nees_full_system / total_dof);
            double log_variance = (variance_nees_full_system > 0) ? 
                                 std::log(variance_nees_full_system / (2.0 * total_dof)) : 
                                 0.0; // Avoid log(0)
            double CNEES = std::abs(log_mean) + std::abs(log_variance);

            // Objective for BO: CNEES
            trials_.push_back({V0, meas_noise_var, CNEES});
            std::cout << "V0: " << V0 << ", meas_noise_var: " << meas_noise_var 
                      << ", CNEES: " << CNEES 
                      << ", ε̃_x: " << mean_nees_full_system << " (expected: " << total_dof << ")"
                      << ", S̃_x: " << variance_nees_full_system << " (expected: " << 2*total_dof << ")" << std::endl;
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
            return CNEES;
        }
        trials_.push_back({V0, meas_noise_var, CNEES});
        std::cout << "V0: " << V0 << ", meas_noise_var: " << meas_noise_var 
                  << ", CNEES: " << CNEES 
                  << ", NEES_mean: N/A, NEES_var: N/A (out of bounds)" << std::endl;
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
        return CNEES;
    }
private:
    const std::vector<std::vector<Eigen::Vector4d>> &all_states_;
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements_;
    std::vector<std::array<double, 3>> &trials_;
    double lower_bound_Q_, upper_bound_Q_, lower_bound_R_, upper_bound_R_;
    double dt_;
};

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");

    // Get dt from data generation config
    double dt = config["Data_Generation"]["dt"].as<double>();
    std::cout << "Using dt = " << dt << " from data generation config" << std::endl;

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
        lb(0), ub(0), lb(1), ub(1), dt
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
    YAML::Node out_config = YAML::LoadFile("../BO_Parameters.yaml");
    out_config["validate_filter"]["q"] = result(0);  // V0 (process noise intensity)
    out_config["validate_filter"]["R"] = result(1);  // meas_noise_std² (measurement noise variance)
    out_config["validate_filter"]["min_objective"] = minC;
    std::ofstream yaml_out("../BO_Parameters.yaml");
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
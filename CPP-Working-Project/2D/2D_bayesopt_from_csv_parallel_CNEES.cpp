#include "2D_factor_graph_trajectory.h"
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
        double meas_noise_std = sqrt(query(1));  // Measurement noise standard deviation
        double CNEES = 1e6;
        
        if (V0 >= lower_bound_Q_ && V0 <= upper_bound_Q_ && meas_noise_std*meas_noise_std >= lower_bound_R_ && meas_noise_std*meas_noise_std <= upper_bound_R_) {
            int num_graphs = all_states_.size();
            int T = all_states_[0].size();
            int nx = 4; // state dimension
            
            // Step 1: Compute NEES for each run and time step
            std::vector<std::vector<double>> nees(num_graphs, std::vector<double>(T, 0.0));
            #pragma omp parallel for
            for (int run = 0; run < num_graphs; ++run) {
                FactorGraph2DTrajectory fg;
                
                // Use proper Q matrix structure for 2D linear tracking with theoretical structure
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
                Eigen::MatrixXd infoMat = fg.getFullInformationMatrix();
                
                // Check if information matrix is valid
                if (infoMat.rows() == 0 || infoMat.cols() == 0) {
                    std::cerr << "ERROR: Information matrix is empty for run " << run << std::endl;
                    continue;
                }
                
                // Check if information matrix has the expected size
                if (infoMat.rows() != T * 4 || infoMat.cols() != T * 4) {
                    std::cerr << "ERROR: Information matrix has wrong size: " 
                              << infoMat.rows() << "x" << infoMat.cols() 
                              << " (expected " << T * 4 << "x" << T * 4 << ")" << std::endl;
                    continue;
                }
                
                for (int k = 0; k < T; ++k) {
                    Eigen::Vector4d err = true_states[k] - est_states[k];
                    Eigen::Matrix4d info_block = infoMat.block<4,4>(k*4, k*4);
                    
                    // Debug: Print detailed information for first few iterations
                    if (eval_count <= 3 && run == 0 && k < 3) {
                        std::cout << "DEBUG: Iteration " << eval_count << ", Run " << run << ", Time " << k << std::endl;
                        std::cout << "  True state: " << true_states[k].transpose() << std::endl;
                        std::cout << "  Est state:  " << est_states[k].transpose() << std::endl;
                        std::cout << "  Error:      " << err.transpose() << std::endl;
                        std::cout << "  Error norm: " << err.norm() << std::endl;
                        std::cout << "  Info block:" << std::endl << info_block << std::endl;
                        std::cout << "  Info block trace: " << info_block.trace() << std::endl;
                        std::cout << "  Info block det:   " << info_block.determinant() << std::endl;
                        
                        // Check if info block is positive definite
                        Eigen::LLT<Eigen::Matrix4d> lltOfInfo(info_block);
                        bool is_pd = (lltOfInfo.info() == Eigen::Success);
                        std::cout << "  Info block is PD: " << (is_pd ? "YES" : "NO") << std::endl;
                        
                        if (is_pd) {
                            Eigen::Matrix4d cov_block = info_block.inverse();
                            std::cout << "  Cov block:" << std::endl << cov_block << std::endl;
                            std::cout << "  Cov block trace: " << cov_block.trace() << std::endl;
                        }
                    }
                    
                    double nees_k = err.transpose() * info_block * err;
                    nees[run][k] = nees_k;
                    
                    // Debug: Print NEES value for first few iterations
                    if (eval_count <= 3 && run == 0 && k < 3) {
                        std::cout << "  NEES: " << nees_k << std::endl;
                        std::cout << "  ---" << std::endl;
                    }
                }
            }
            
            // Step 2: For each time step, average NEES over all runs
            std::vector<double> mean_nees_per_timestep(T, 0.0);
            for (int k = 0; k < T; ++k) {
                for (int run = 0; run < num_graphs; ++run) {
                    mean_nees_per_timestep[k] += nees[run][k];
                }
                mean_nees_per_timestep[k] /= num_graphs;
            }
            
            // Step 3: Average over all time steps
            double mean_nees = 0.0;
            for (int k = 0; k < T; ++k) mean_nees += mean_nees_per_timestep[k];
            mean_nees /= T;
            
            // Step 4: Compute S_x (variance of normalized NEES across all runs and time steps)
            double Sx = 0.0;
            for (int k = 0; k < T; ++k) {
                for (int run = 0; run < num_graphs; ++run) {
                    double normed = nees[run][k] / nx;
                    double normed_mean = mean_nees_per_timestep[k] / nx;
                    Sx += (normed - normed_mean) * (normed - normed_mean);
                }
            }
            if (num_graphs > 1) {
                Sx = Sx / (T * (num_graphs - 1));
            } else {
                Sx = 0.0; // Avoid division by zero if only one run
            }

            // Compute CNEES for reporting
            double log_mean = std::log(mean_nees / nx);
            double log_Sx = (Sx > 0) ? std::log(Sx / (2 * nx)) : 0.0; // Avoid log(0)
            double CNEES = std::abs(log_mean) + std::abs(log_Sx);

            // Calculate overall NEES statistics across all runs and time steps
            std::vector<double> all_nees_values;
            all_nees_values.reserve(num_graphs * T);
            for (int run = 0; run < num_graphs; ++run) {
                for (int k = 0; k < T; ++k) {
                    all_nees_values.push_back(nees[run][k]);
                }
            }
            
            // Calculate mean and variance of all NEES values
            double overall_mean = 0.0;
            for (const auto& val : all_nees_values) {
                overall_mean += val;
            }
            overall_mean /= all_nees_values.size();
            
            double overall_variance = 0.0;
            for (const auto& val : all_nees_values) {
                overall_variance += (val - overall_mean) * (val - overall_mean);
            }
            overall_variance /= (all_nees_values.size() - 1);

            // Objective for BO: CNEES
            trials_.push_back({V0, meas_noise_std*meas_noise_std, CNEES});
            std::cout << "V0: " << V0 << ", meas_noise_std²: " << meas_noise_std*meas_noise_std 
                      << ", CNEES: " << CNEES 
                      << ", NEES_mean: " << overall_mean << ", NEES_var: " << overall_variance << std::endl;
            auto eval_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eval_duration = eval_end - eval_start;
            std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
            return CNEES;
        }
        trials_.push_back({V0, meas_noise_std*meas_noise_std, CNEES});
        std::cout << "V0: " << V0 << ", meas_noise_std²: " << meas_noise_std*meas_noise_std 
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
    infer_problem_size_h5("../H5_Files/2D_noisy_states.h5", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from HDF5." << std::endl;
        return 1;
    }
    std::cout << "Detected " << num_graphs << " runs, each of length " << Trajectory_length << std::endl;

    // Load all runs' data from HDF5
    auto all_states = load_all_noisy_states_h5("../H5_Files/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../H5_Files/2D_noisy_measurements.h5");

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
    const std::string h5_filename = "../H5_Files/2D_bayesopt_trials.h5";
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

    std::cout << "Best V0: " << result(0) << ", Best meas_noise_std²: " << result(1) 
              << ", Best CNEES: " << minC << std::endl;
    std::cout << "[Timing] Total BayesOpt optimization took " << total_duration.count() << " seconds." << std::endl;
    std::ofstream cfile("../H5_Files/2D_bayesopt_best.txt");
    cfile << "Best V0: " << result(0) << ", Best meas_noise_std²: " << result(1) 
          << ", Best CNEES: " << minC << std::endl;
    cfile.close();

    return 0;
} 
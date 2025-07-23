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
#include <fstream>

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
                    std::vector<std::array<double, 3>> &trials)
        : bayesopt::ContinuousModel(2, params),
          all_states_(all_states),
          all_measurements_(all_measurements),
          trials_(trials) {}

    double evaluateSample(const boost::numeric::ublas::vector<double> &query) override {
        static int eval_count = 0;
        eval_count++;
        std::cout << "[BayesOpt] Iteration: " << eval_count << std::endl;
        auto eval_start = std::chrono::high_resolution_clock::now();
        double Qval = query(0);
        double Rval = query(1);
        const int d = 4; // state dimension
        std::vector<double> all_nees;
        int num_graphs = all_states_.size();

        #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            FactorGraph2DTrajectory fg;
            fg.Q_ = Eigen::Matrix4d::Identity() * Qval;
            fg.R_ = Eigen::Matrix2d::Identity() * Rval;
            fg.run(all_states_[run], &all_measurements_[run], false);
            int num_estimates = fg.getAllEstimates().size();
            std::vector<double> run_nees(num_estimates);
            for (int k = 0; k < num_estimates; ++k) {
                Eigen::Vector4d true_state = all_states_[run][k];
                Eigen::Vector4d estimated_state = fg.getEstimate(k);
                Eigen::Vector4d error = estimated_state - true_state;
                Eigen::MatrixXd H = fg.getFullInformationMatrix();
                Eigen::Matrix4d P_inv_k = H.block<4,4>(k*4, k*4);
                run_nees[k] = error.transpose() * P_inv_k * error;
            }
            #pragma omp critical
            all_nees.insert(all_nees.end(), run_nees.begin(), run_nees.end());
        }

        // Compute mean and variance of NEES
        double mean_nees = 0.0;
        for (double v : all_nees) mean_nees += v;
        mean_nees /= all_nees.size();
        double var_nees = 0.0;
        for (double v : all_nees) var_nees += (v - mean_nees) * (v - mean_nees);
        var_nees /= (all_nees.size() - 1);

        // Objective: squared error from theoretical mean and variance
        double obj = (mean_nees - d) * (mean_nees - d) + (var_nees - 2*d) * (var_nees - 2*d);
        trials_.push_back({Qval, Rval, obj});

        std::cout << "Q: " << Qval << ", R: " << Rval << ", mean NEES: " << mean_nees << ", var NEES: " << var_nees << ", Obj: " << obj << std::endl;
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;

        return obj;
    }

private:
    const std::vector<std::vector<Eigen::Vector4d>> &all_states_;
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements_;
    std::vector<std::array<double, 3>> &trials_;
};

int main() {
    // Load configuration from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");

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

    // Bounds for Q and R from YAML
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = config["parameters"][0]["lower_bound"].as<double>();
    ub(0) = config["parameters"][0]["upper_bound"].as<double>();
    lb(1) = config["parameters"][1]["lower_bound"].as<double>();
    ub(1) = config["parameters"][1]["upper_bound"].as<double>();

    CMetricBayesOpt opt(params, all_states, all_measurements, trials);
    opt.setBoundingBox(lb, ub);
    boost::numeric::ublas::vector<double> result(2);
    auto total_start = std::chrono::high_resolution_clock::now();
    opt.optimize(result);
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    double min_objective = opt.getValueAtMinimum();

    // Save best Q, R, and final objective value to BO_Parameters.yaml
    YAML::Node out_config = YAML::LoadFile("../BO_Parameters.yaml");
    out_config["validate_filter"]["Q"] = result(0);
    out_config["validate_filter"]["R"] = result(1);
    out_config["validate_filter"]["min_objective"] = min_objective;
    std::ofstream yaml_out("../BO_Parameters.yaml");
    yaml_out << out_config;
    yaml_out.close();

    std::cout << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Min Objective: " << min_objective << std::endl;
    std::cout << "[Timing] Total BayesOpt optimization took " << total_duration.count() << " seconds." << std::endl;

    return 0;
} 
#include "2D_factor_graph_trajectory.h"
#include "2D_h5_loader.h"
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include "H5Cpp.h"
#include <array>

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
        auto eval_start = std::chrono::high_resolution_clock::now();
        double Qval = query(0);
        double Rval = query(1);
        double C = 1e6;
        // Enforce bounds for safety
        if (Qval >= 0.1 && Rval >= 0.1) {
            int num_graphs = all_states_.size();
            int Trajectory_length = all_states_[0].size();
            std::vector<double> chi2_values(num_graphs);
            for (int run = 0; run < num_graphs; ++run) {
                FactorGraph2DTrajectory fg;
                fg.Q_ = Eigen::Matrix4d::Identity() * Qval;
                fg.R_ = Eigen::Matrix2d::Identity() * Rval;
                fg.run(all_states_[run], &all_measurements_[run], false, 1.0);  // Use dt=1.0 for this file
                chi2_values[run] = fg.getChi2();
            }
            double meanChi2 = 0.0;
            for (int i = 0; i < num_graphs; ++i) meanChi2 += chi2_values[i];
            meanChi2 /= num_graphs;
            double covChi2 = 0.0;
            for (int i = 0; i < num_graphs; ++i) covChi2 += (chi2_values[i] - meanChi2) * (chi2_values[i] - meanChi2);
            covChi2 /= (num_graphs - 1);
            int N = static_cast<int>(all_measurements_[0].size()) * 2 + (static_cast<int>(all_states_[0].size()) - 1) * 4;
            C = std::abs(std::log(meanChi2 / N)) + std::abs(std::log(covChi2 / (2.0 * N)));
        }
        trials_.push_back({Qval, Rval, C});
        std::cout << "Q: " << Qval << ", R: " << Rval << ", C: " << C << std::endl;
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
        return C;
    }
private:
    const std::vector<std::vector<Eigen::Vector4d>> &all_states_;
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements_;
    std::vector<std::array<double, 3>> &trials_;
};

int main() {
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
    params.n_iterations = 10;
    params.n_init_samples = 1;
    params.crit_name = (char*)"cEI";
    params.verbose_level = 1;
    params.surr_name = (char*)"sGaussianProcess";
    params.noise = 1e-6;

    // Bounds for Q and R
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = 0.1; lb(1) = 0.1;
    ub(0) = 1.0;  ub(1) = 1.0;

    CMetricBayesOpt opt(params, all_states, all_measurements, trials);
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

    std::cout << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best C: " << minC << std::endl;
    std::cout << "[Timing] Total BayesOpt optimization took " << total_duration.count() << " seconds." << std::endl;
    std::ofstream cfile("../H5_Files/2D_bayesopt_best.txt");
    cfile << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best C: " << minC << std::endl;
    cfile.close();

    return 0;
} 
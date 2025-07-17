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

// Define Hyper Parameters for BAYESOPT
void initialisation(bopt_params &params) {
    params.n_iterations = 125;
    params.n_init_samples = 125;
    params.crit_name = (char*)"cEI"; // Expected Improvement
    params.verbose_level = 1;
    params.surr_name = (char*)"sGaussianProcess";  
    params.noise = 1e-6;
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
        double CNEES = 1e6;
        if (Qval >= 0.1 && Rval >= 0.1) {
            int num_graphs = all_states_.size();
            int T = all_states_[0].size();
            int nx = 4; // state dimension
            // Step 1: Compute NEES for each run and time step
            std::vector<std::vector<double>> nees(num_graphs, std::vector<double>(T, 0.0));
            #pragma omp parallel for
            for (int run = 0; run < num_graphs; ++run) {
                FactorGraph2DTrajectory fg;
                fg.Q_ = Eigen::Matrix4d::Identity() * Qval;
                fg.R_ = Eigen::Matrix2d::Identity() * Rval;
                FactorGraph2DTrajectory::OutputOptions opts;
                opts.output_estimated_state = true;
                opts.output_true_state = true;
                opts.output_information_matrix = true;
                fg.setOutputOptions(opts);
                fg.run(all_states_[run], &all_measurements_[run], false);
                auto est_states = fg.getAllEstimates();
                auto true_states = fg.getAllTrueStates();
                Eigen::MatrixXd infoMat = fg.getFullInformationMatrix();
                for (int k = 0; k < T; ++k) {
                    Eigen::Vector4d err = true_states[k] - est_states[k];
                    Eigen::Matrix4d info_block = infoMat.block<4,4>(k*4, k*4);
                    double nees_k = err.transpose() * info_block * err;
                    nees[run][k] = nees_k;
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
            // Step 4: Compute S_x (standard deviation of normalized NEES across all runs and time steps)
            double Sx = 0.0;
            for (int k = 0; k < T; ++k) {
                for (int run = 0; run < num_graphs; ++run) {
                    double normed = nees[run][k] / nx;
                    double normed_mean = mean_nees_per_timestep[k] / nx;
                    Sx += (normed - normed_mean) * (normed - normed_mean);
                }
            }
            if (num_graphs > 1) {
                Sx = std::sqrt(Sx / (T * (num_graphs - 1)));
            } else {
                Sx = 0.0; // Avoid division by zero if only one run
            }

            // Step 5: Compute the augmented CNEES metric
            double log_mean = std::log(mean_nees / nx);
            double log_Sx = (Sx > 0) ? std::log(Sx / (2*nx)) : 0.0; // Avoid log(0)
            CNEES = std::sqrt(log_mean * log_mean + log_Sx * log_Sx);
        }
        trials_.push_back({Qval, Rval, CNEES});
        std::cout << "Q: " << Qval << ", R: " << Rval << ", CNEES: " << CNEES << std::endl;
        auto eval_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eval_duration = eval_end - eval_start;
        std::cout << "[Timing] evaluateSample took " << eval_duration.count() << " seconds." << std::endl;
        return CNEES;
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
    initialisation(params);

    // Bounds for Q and R
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = 0.1; lb(1) = 0.1;
    ub(0) = 5.0;  ub(1) = 5.0;

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

    std::cout << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best CNEES: " << minC << std::endl;
    std::cout << "[Timing] Total BayesOpt optimization took " << total_duration.count() << " seconds." << std::endl;
    std::ofstream cfile("../H5_Files/2D_bayesopt_best.txt");
    cfile << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best CNEES: " << minC << std::endl;
    cfile.close();

    return 0;
} 
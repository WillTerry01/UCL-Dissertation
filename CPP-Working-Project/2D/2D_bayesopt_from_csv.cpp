#include "2D_factor_graph_trajectory.h"
#include "2D_csv_loader.h"
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.h>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include <limits>

// Helper function to infer num_graphs and N from the noisy states CSV
void infer_problem_size(const std::string& filename, int& num_graphs, int& Trajectory_length) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        num_graphs = 0;
        Trajectory_length = 0;
        return;
    }
    std::string line;
    std::getline(file, line); // skip header
    std::set<int> runs;
    int max_t = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        int run, t;
        std::getline(ss, item, ','); run = std::stoi(item);
        std::getline(ss, item, ','); t = std::stoi(item);
        runs.insert(run);
        if (t > max_t) max_t = t;
    }
    num_graphs = runs.size();
    Trajectory_length = max_t + 1;
}

class CMetricBayesOpt : public bayesopt::ContinuousModel {
public:
    CMetricBayesOpt(const bopt_params &params,
                    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
                    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements)
        : bayesopt::ContinuousModel(2, params),
          all_states_(all_states),
          all_measurements_(all_measurements) {}

    double evaluateSample(const boost::numeric::ublas::vector<double> &query) override {
        double Qval = query(0);
        double Rval = query(1);
        // Enforce bounds for safety
        if (Qval < 0.1 || Rval < 0.1) {
            // Log the invalid trial
            std::ofstream trialfile("../2D/2D_bayesopt_trials.csv", std::ios::app);
            trialfile << Qval << "," << Rval << "," << 1e6 << "\n";
            trialfile.close();
            return 1e6;
        }
        int num_graphs = all_states_.size();
        int Trajectory_length = all_states_[0].size();
        std::vector<double> chi2_values(num_graphs);
        for (int run = 0; run < num_graphs; ++run) {
            FactorGraph2DTrajectory fg;
            fg.Q_ = Eigen::Matrix4d::Identity() * Qval;
            fg.R_ = Eigen::Matrix2d::Identity() * Rval;
            fg.run(all_states_[run], &all_measurements_[run], false);
            chi2_values[run] = fg.getChi2();
        }
        // Compute meanChi2
        double meanChi2 = 0.0;
        for (int i = 0; i < num_graphs; ++i) meanChi2 += chi2_values[i];
        meanChi2 /= num_graphs;
        // Compute covChi2
        double covChi2 = 0.0;
        for (int i = 0; i < num_graphs; ++i) covChi2 += (chi2_values[i] - meanChi2) * (chi2_values[i] - meanChi2);
        covChi2 /= (num_graphs - 1);
        // Degrees of freedom: (N-1)*4 (process) + N*2 (measurement)
        int N = static_cast<int>(all_measurements_[0].size()) * 2 + (static_cast<int>(all_states_[0].size()) - 1) * 4;
        double C = std::abs(std::log(meanChi2 / N)) + std::abs(std::log(covChi2 / (2.0 * N)));
        std::cout << "Q: " << Qval << ", R: " << Rval << ", C: " << C << std::endl;
        // Log the trial
        std::ofstream trialfile("../2D/2D_bayesopt_trials.csv", std::ios::app);
        trialfile << Qval << "," << Rval << "," << C << "\n";
        trialfile.close();
        return C;
    }
private:
    const std::vector<std::vector<Eigen::Vector4d>> &all_states_;
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements_;
};

int main() {
    // Write header for trials CSV (before any optimization or evaluation)
    std::ofstream trialfile("../2D/2D_bayesopt_trials.csv", std::ios::trunc);
    trialfile << "Q,R,C\n";
    trialfile.close();

    // Infer problem size from the noisy states CSV
    int Trajectory_length = 0;
    int num_graphs = 0;
    infer_problem_size("../2D/2D_noisy_states.csv", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from CSV." << std::endl;
        return 1;
    }
    std::cout << "Detected " << num_graphs << " runs, each of length " << Trajectory_length << std::endl;

    // Load all runs' data
    auto all_states = load_all_noisy_states("../2D/2D_noisy_states.csv", Trajectory_length);
    auto all_measurements = load_all_noisy_measurements("../2D/2D_noisy_measurements.csv", Trajectory_length);

    // Set up BayesOpt parameters
    bopt_params params = initialize_parameters_to_default();
    params.n_iterations = 500;
    params.n_init_samples = 50;
    params.crit_name = (char*)"cEI"; // Expected Improvement
    params.verbose_level = 1;
    params.surr_name = (char*)"sGaussianProcess";
    params.noise = 1e-6;

    // Bounds for Q and R
    boost::numeric::ublas::vector<double> lb(2), ub(2);
    lb(0) = 0.1; lb(1) = 0.1;
    ub(0) = 1.0;  ub(1) = 1.0;

    CMetricBayesOpt opt(params, all_states, all_measurements);
    boost::numeric::ublas::vector<double> result(2);
    opt.optimize(result);
    double minC = opt.getValueAtMinimum();

    std::cout << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best C: " << minC << std::endl;
    std::ofstream cfile("../2D/2D_bayesopt_best.txt");
    cfile << "Best Q: " << result(0) << ", Best R: " << result(1) << ", Best C: " << minC << std::endl;
    cfile.close();

    return 0;
} 
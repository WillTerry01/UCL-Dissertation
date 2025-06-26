#include "2D_factor_graph_trajectory.h"
#include "2D_csv_loader.h"
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <set>

// Helper function to infer num_graphs and N from the noisy states CSV
void infer_problem_size(const std::string& filename, int& num_graphs, int& N) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        num_graphs = 0;
        N = 0;
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
    N = max_t + 1;
}

int main() {
    std::cout << "got here" << std::endl;

    // Infer problem size from the noisy states CSV
    int Trajectory_length = 0; // Trajectory length
    int num_graphs = 0; // Number of Monte Carlo samples
    infer_problem_size("../2D/2D_noisy_states.csv", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from CSV." << std::endl;
        return 1;
    }
    std::cout << "Detected " << num_graphs << " runs, each of length " << Trajectory_length << std::endl;

    // Load all runs' data
    auto all_states = load_all_noisy_states("../2D/2D_noisy_states.csv", Trajectory_length);
    auto all_measurements = load_all_noisy_measurements("../2D/2D_noisy_measurements.csv", Trajectory_length);

    if (all_states.size() != num_graphs || all_measurements.size() != num_graphs) {
        std::cerr << "Error: Number of runs in CSV does not match detected num_graphs." << std::endl;
        return 1;
    }

    std::vector<double> chi2_values(num_graphs);

    // Open summary CSV
    std::ofstream chi2_csv("../2D/2D_chi2_results.csv");
    chi2_csv << "run,chi2\n";

    // Open all trajectory estimates CSV
    std::ofstream all_traj_csv("../2D/2D_trajectory_estimates_all.csv");
    all_traj_csv << "run,t,true_x,true_y,meas_x,meas_y,est_x,est_y\n";

    for (int run = 0; run < num_graphs; ++run) {
        std::cout << "Run " << run << " / " << num_graphs << std::endl;
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * 0.5; // Tune as desired
        fg.R_ = Eigen::Matrix2d::Identity() * 0.5; // Tune as desired
        fg.run(all_states[run], &all_measurements[run], false);
        chi2_values[run] = fg.getChi2();
        chi2_csv << run << "," << chi2_values[run] << "\n";
        // Save all trajectory estimates to one file
        for (int t = 0; t < Trajectory_length; ++t) {
            const auto& true_state = all_states[run][t];
            const auto& meas = all_measurements[run][t];
            const auto& v = fg.getEstimate(t);
            all_traj_csv << run << "," << t << ","
                         << true_state[0] << "," << true_state[1] << ","
                         << meas[0] << "," << meas[1] << ","
                         << v[0] << "," << v[1] << "\n";
        }
    }
    chi2_csv.close();
    all_traj_csv.close();
    std::cout << "Saved chi2 values for " << num_graphs << " runs to ../2D/2D_chi2_results.csv" << std::endl;
    std::cout << "Saved all trajectory estimates to ../2D/2D_trajectory_estimates_all.csv" << std::endl;

    // Compute the CNEES/CNIS consistency metric
    double meanChi2 = 0.0;
    double covChi2 = 0.0;
    int N = 0;
    // Compute meanChi2
    for (int i = 0; i < num_graphs; ++i) meanChi2 += chi2_values[i];
    meanChi2 /= num_graphs;
    // Compute covChi2
    for (int i = 0; i < num_graphs; ++i) covChi2 += (chi2_values[i] - meanChi2) * (chi2_values[i] - meanChi2);
    covChi2 /= (num_graphs - 1);
    // Degrees of freedom: number of residuals per run (from the first run)
    // We can estimate as the number of measurements + number of process edges
    // For your setup, it's (N-1)*4 (process) + N*2 (measurement) = 4*(N-1) + 2*N = 6N-4
    N = static_cast<int>(all_measurements[0].size()) * 2 + (static_cast<int>(all_states[0].size()) - 1) * 4;

    double C = std::abs(std::log(meanChi2 / N)) + std::abs(std::log(covChi2 / (2.0 * N)));

    std::cout << "Consistency metric C: " << C << std::endl;
    std::ofstream cfile("../2D/2D_consistency_metric.txt");
    cfile << "C = " << C << std::endl;
    cfile.close();
    std::cout << "Saved consistency metric to ../2D/2D_consistency_metric.txt" << std::endl;

    return 0;
} 
#include "1D_factor_graph_trajectory.h"
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

int main() {
    int N = 20;
    int num_runs = 1000;
    std::vector<double> chi2_values(num_runs);

    // Calculate the trajectory to run all runs off
    std::vector<Eigen::Vector2d> true_states(N);
    double pos = 0.0;
    double vel = 1.0;
    double dt = 1.0;
    true_states[0] << pos, vel;
    for (int k = 1; k < N; ++k) {
        true_states[k][0] = true_states[k-1][0] + true_states[k-1][1] * dt;
        true_states[k][1] = true_states[k-1][1];
    }

    for (int run = 0; run < num_runs; ++run) {
        if (run % 100 == 0) {
            std::cout << "Run " << run << " / " << num_runs << std::endl;
        }
        FactorGraph1DTrajectory fg;
        fg.Q_ = Eigen::Matrix2d::Identity() * 0.1;
        fg.R_ = Eigen::Matrix<double, 1, 1>::Identity() * 0.2;
        fg.run(true_states, nullptr, true); // Add process noise inside the function
        chi2_values[run] = fg.getChi2();
    }

    // Write chi2 values to CSV
    std::ofstream csv("../1D/1D_chi2_results.csv");
    csv << "run,chi2\n";
    for (int run = 0; run < num_runs; ++run) {
        csv << run << "," << chi2_values[run] << "\n";
    }
    csv.close();
    std::cout << "Saved chi2 values for " << num_runs << " runs to 1D_chi2_results.csv" << std::endl;
    return 0;
} 
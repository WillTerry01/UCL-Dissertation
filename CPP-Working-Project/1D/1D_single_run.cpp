#include "1D_factor_graph_trajectory.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Example main function for user-provided trajectory
int main() {
    int N = 20;
    std::vector<Eigen::Vector2d> true_states(N);
    double pos = 0.0;
    double vel = 1.0;
    double dt = 1.0;
    true_states[0] << pos, vel;
    for (int k = 1; k < N; ++k) {
        true_states[k][0] = true_states[k-1][0] + true_states[k-1][1] * dt;
        true_states[k][1] = true_states[k-1][1];
    }
    FactorGraph1DTrajectory fg;
    fg.Q_ = Eigen::Matrix2d::Identity() * 1.0;
    fg.R_ = Eigen::Matrix<double, 1, 1>::Identity() * 1.0;
    fg.run(true_states, nullptr, true); // Add process noise inside the function
    fg.writeCSV("../1D/1d_trajectory_estimate.csv");
    std::cout << "Final chi2: " << fg.getChi2() << std::endl;
    return 0;
} 
#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Example main function for user-provided trajectory
int main() {
    // Example: straight line trajectory, process noise added by the factor graph class
    int N = 20;
    std::vector<Eigen::Vector4d> true_states(N);
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    double dt = 1.0;
    true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
    for (int k = 1; k < N; ++k) {
        true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
        true_states[k].tail<2>() = true_states[k-1].tail<2>();
    }
    FactorGraph2DTrajectory fg;
    fg.Q_ = Eigen::Matrix4d::Identity() * 0.01;
    fg.R_ = Eigen::Matrix2d::Identity() * 0.01;
    // Optionally modify fg.Q_ and fg.R_ here
    fg.run(true_states, nullptr, true); // Add process noise inside the function
    fg.printHessian();
    // fg.writeCSV("../H5_Files/2D_trajectory_estimate.csv");
    fg.writeHDF5("../H5_Files/2D_trajectory_estimate.h5");
    std::cout << "Final chi2: " << fg.getChi2() << std::endl;
    return 0;
} 
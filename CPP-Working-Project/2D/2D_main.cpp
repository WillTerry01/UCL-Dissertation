#include "2D_factor_graph_trajectory.h"
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

int main() {
    int N = 20;
    int num_runs = 1000;
    std::vector<double> chi2_values(num_runs);

    //calculate the trajectory to run all runs off
    std::vector<Eigen::Vector4d> true_states(N);
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    double dt = 1.0;
    true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
    for (int k = 1; k < N; ++k) {
        true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
        true_states[k].tail<2>() = true_states[k-1].tail<2>();
        
    }
    /*
    std::cout << "true_states:" << std::endl;
    for (int k = 0; k < N; ++k) {
        std::cout << "  [" << k << "]: " << true_states[k].transpose() << std::endl;
    }
    */

    for (int run = 0; run < num_runs; ++run) {
        if (run % 100 == 0) {
            std::cout << "Run " << run << " / " << num_runs << std::endl;
        }
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * 0.110528;
        fg.R_ = Eigen::Matrix2d::Identity() * 0.134589;
        fg.run(true_states, nullptr, true); // Add process noise inside the function
        chi2_values[run] = fg.getChi2();
    }

    // Write chi2 values to CSV
    std::ofstream csv("../2D/2D_chi2_results.csv");
    csv << "run,chi2\n";
    for (int run = 0; run < num_runs; ++run) {
        csv << run << "," << chi2_values[run] << "\n";
    }
    csv.close();
    std::cout << "Saved chi2 values for " << num_runs << " runs to chi2_results.csv" << std::endl;
    return 0;
}

#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <random>

int main() {
    int N = 20; // Trajectory length
    int num_runs = 100; // Number of Monte Carlo runs
    double dt = 1.0;
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    double Qval = 0.1; // Process noise
    double Rval = 0.2; // Measurement noise
    unsigned int base_seed = 42;

    std::ofstream results_csv("2D_single_run_mc_results.csv");
    results_csv << "run,chi2,mse\n";

    for (int run = 0; run < num_runs; ++run) {
        // Generate true trajectory
        std::vector<Eigen::Vector4d> true_states(N);
        true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        for (int k = 1; k < N; ++k) {
            true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
            true_states[k].tail<2>() = true_states[k-1].tail<2>();
        }

        // Add process noise
        std::mt19937 gen(base_seed + run);
        std::normal_distribution<> noise_q(0.0, std::sqrt(Qval));
        std::vector<Eigen::Vector4d> noisy_states = true_states;
        for (int k = 1; k < N; ++k) {
            Eigen::Vector4d process_noise;
            for (int i = 0; i < 4; ++i) process_noise[i] = noise_q(gen);
            noisy_states[k] += process_noise;
        }

        // Generate noisy measurements
        std::normal_distribution<> noise_r(0.0, std::sqrt(Rval));
        std::vector<Eigen::Vector2d> noisy_measurements(N);
        for (int k = 0; k < N; ++k) {
            noisy_measurements[k][0] = noisy_states[k][0] + noise_r(gen);
            noisy_measurements[k][1] = noisy_states[k][1] + noise_r(gen);
        }

        // Run the factor graph
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * Qval;
        fg.R_ = Eigen::Matrix2d::Identity() * Rval;
        fg.run(noisy_states, &noisy_measurements, false);
        double chi2 = fg.getChi2();

        // Compute MSE between estimated and true trajectory (positions only)
        double mse = 0.0;
        for (int k = 0; k < N; ++k) {
            Eigen::Vector2d est_pos = fg.getEstimate(k).head<2>();
            Eigen::Vector2d true_pos = true_states[k].head<2>();
            mse += (est_pos - true_pos).squaredNorm();
        }
        mse /= N;

        results_csv << run << "," << chi2 << "," << mse << "\n";
    }
    results_csv.close();
    std::cout << "Saved results to 2D_single_run_mc_results.csv" << std::endl;
    return 0;
} 
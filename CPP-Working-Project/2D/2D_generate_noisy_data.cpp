#include <Eigen/Dense>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    // Parameters
    int N = 20; // Trajectory length
    int num_graphs = 100; // Number of Monte Carlo samples
    double dt = 1.0;
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    // Noise parameters (can be changed as needed)
    double process_noise_std = 0.3162; // sqrt(0.1)
    double meas_noise_std = 0.4472;    // sqrt(0.2)
    unsigned int base_seed = 42; // Fixed base seed for repeatability

    // Open output files and write headers
    std::ofstream states_csv("../2D/2D_noisy_states.csv");
    std::ofstream meas_csv("../2D/2D_noisy_measurements.csv");
    states_csv << "run,t,x,y,vx,vy\n";
    meas_csv << "run,t,x_meas,y_meas\n";

    for (int run = 0; run < num_graphs; ++run) {
        // True trajectory
        std::vector<Eigen::Vector4d> true_states(N);
        true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
        for (int k = 1; k < N; ++k) {
            true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
            true_states[k].tail<2>() = true_states[k-1].tail<2>();
        }

        // Add process noise to states
        std::mt19937 gen(base_seed + run); // Different seed for each run
        std::normal_distribution<> noise_q(0.0, process_noise_std);
        std::vector<Eigen::Vector4d> noisy_states = true_states;
        for (int k = 1; k < N; ++k) {
            Eigen::Vector4d process_noise;
            for (int i = 0; i < 4; ++i) process_noise[i] = noise_q(gen);
            noisy_states[k] += process_noise;
        }

        // Generate noisy measurements from noisy states
        std::normal_distribution<> noise_r(0.0, meas_noise_std);
        std::vector<Eigen::Vector2d> noisy_measurements(N);
        for (int k = 0; k < N; ++k) {
            noisy_measurements[k][0] = noisy_states[k][0] + noise_r(gen);
            noisy_measurements[k][1] = noisy_states[k][1] + noise_r(gen);
        }

        // Write to CSVs
        for (int k = 0; k < N; ++k) {
            states_csv << run << "," << k << "," << noisy_states[k][0] << "," << noisy_states[k][1] << "," << noisy_states[k][2] << "," << noisy_states[k][3] << "\n";
            meas_csv << run << "," << k << "," << noisy_measurements[k][0] << "," << noisy_measurements[k][1] << "\n";
        }
    }
    states_csv.close();
    meas_csv.close();
    std::cout << "Saved all noisy states to ../2D/2D_noisy_states.csv and all measurements to ../2D/2D_noisy_measurements.csv" << std::endl;
    return 0;
} 
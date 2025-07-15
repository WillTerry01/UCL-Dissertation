#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include "H5Cpp.h"

int main() {
    // Parameters
    int N = 20; // Trajectory length
    int num_graphs = 1000; // Number of Monte Carlo samples
    double dt = 1.0;
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    // Noise parameters (can be changed as needed)
    double process_noise_std = sqrt(0.25); // sqrt(0.1)
    double meas_noise_std = sqrt(0.25);    // sqrt(0.2)
    unsigned int base_seed = 42; // Fixed base seed for repeatability

    // Prepare output arrays
    std::vector<double> states(num_graphs * N * 4);
    std::vector<double> measurements(num_graphs * N * 2);

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

        // Store in output arrays
        for (int k = 0; k < N; ++k) {
            int state_idx = run * N * 4 + k * 4;
            states[state_idx + 0] = noisy_states[k][0];
            states[state_idx + 1] = noisy_states[k][1];
            states[state_idx + 2] = noisy_states[k][2];
            states[state_idx + 3] = noisy_states[k][3];
            int meas_idx = run * N * 2 + k * 2;
            measurements[meas_idx + 0] = noisy_measurements[k][0];
            measurements[meas_idx + 1] = noisy_measurements[k][1];
        }
    }

    // Save to HDF5
    const std::string states_h5 = "../H5_Files/2D_noisy_states.h5";
    const std::string meas_h5 = "../H5_Files/2D_noisy_measurements.h5";
    hsize_t states_dims[3] = {static_cast<hsize_t>(num_graphs), static_cast<hsize_t>(N), 4};
    hsize_t meas_dims[3] = {static_cast<hsize_t>(num_graphs), static_cast<hsize_t>(N), 2};

    // States
    H5::H5File states_file(states_h5, H5F_ACC_TRUNC);
    H5::DataSpace states_space(3, states_dims);
    H5::DataSet states_dataset = states_file.createDataSet("states", H5::PredType::NATIVE_DOUBLE, states_space);
    states_dataset.write(states.data(), H5::PredType::NATIVE_DOUBLE);

    // Measurements
    H5::H5File meas_file(meas_h5, H5F_ACC_TRUNC);
    H5::DataSpace meas_space(3, meas_dims);
    H5::DataSet meas_dataset = meas_file.createDataSet("measurements", H5::PredType::NATIVE_DOUBLE, meas_space);
    meas_dataset.write(measurements.data(), H5::PredType::NATIVE_DOUBLE);

    std::cout << "Saved all noisy states to " << states_h5 << " and all measurements to " << meas_h5 << std::endl;
    return 0;
} 
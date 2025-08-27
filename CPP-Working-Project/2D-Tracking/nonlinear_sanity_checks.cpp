#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <yaml-cpp/yaml.h>
#include "2D_h5_loader.h"

static double compute_rmse_axis(const std::vector<Eigen::Vector2d>& meas,
                                const std::vector<Eigen::Vector4d>& states,
                                int axis /*0:x,1:y*/) {
    const int N = std::min(meas.size(), states.size());
    if (N == 0) return 0.0;
    double sumsq = 0.0;
    for (int k = 0; k < N; ++k) {
        double diff = meas[k][axis] - states[k][axis];
        sumsq += diff * diff;
    }
    return std::sqrt(sumsq / N);
}

int main() {
    try {
        YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");
        double dt_default = config["Data_Generation"]["dt"].as<double>();
        double meas_noise_var = config["Data_Generation"]["meas_noise_var"].as<double>();
        double meas_noise_std = std::sqrt(meas_noise_var);

        // Build dt_vec if dt_pieces present
        int N = config["Data_Generation"]["trajectory_length"].as<int>();
        std::vector<double> dt_vec(std::max(0, N - 1), dt_default);
        if (config["Data_Generation"]["dt_pieces"]) {
            for (const auto& piece : config["Data_Generation"]["dt_pieces"]) {
                int from = piece["from"].as<int>();
                int to = piece["to"].as<int>();
                double dt_piece = piece["dt"].as<double>();
                from = std::max(0, from);
                to = std::min(N - 2, to);
                for (int k = from; k <= to; ++k) dt_vec[k] = dt_piece;
            }
        }

        std::cout << "=== Nonlinear Sanity Checks ===\n";
        std::cout << "Default dt: " << dt_default << "\n";
        if (!dt_vec.empty()) {
            std::cout << "dt_vec size: " << dt_vec.size() << ", first 10: ";
            for (size_t i = 0; i < std::min<size_t>(10, dt_vec.size()); ++i) std::cout << dt_vec[i] << (i+1<10?", ":"\n");
            double max_ratio = 1.0;
            for (size_t i = 1; i < dt_vec.size(); ++i) {
                if (dt_vec[i-1] > 0) {
                    max_ratio = std::max(max_ratio, dt_vec[i] / dt_vec[i-1]);
                }
            }
            std::cout << "Max consecutive dt ratio: " << max_ratio << (max_ratio > 2.0 ? " (WARNING: >2)" : "") << "\n";
        }

        // Load data
        auto all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_nonlinear_states.h5");
        auto all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5");
        if (all_states.empty() || all_measurements.empty()) {
            std::cerr << "No data found. Generate nonlinear data first." << std::endl;
            return 1;
        }

        // Compute RMSE for run 0
        const auto& states0 = all_states[0];
        const auto& meas0 = all_measurements[0];
        double rmse_x = compute_rmse_axis(meas0, states0, 0);
        double rmse_y = compute_rmse_axis(meas0, states0, 1);
        double rmse_xy = std::sqrt((rmse_x*rmse_x + rmse_y*rmse_y) / 2.0);
        std::cout << "RMSE x: " << rmse_x << ", RMSE y: " << rmse_y << ", avg RMSE: " << rmse_xy << "\n";
        std::cout << "Expected per-axis sigma ~ meas_noise_std: " << meas_noise_std << "\n";
        if (rmse_x > 3*meas_noise_std || rmse_y > 3*meas_noise_std) {
            std::cout << "WARNING: RMSE significantly larger than measurement std; check data alignment and files." << std::endl;
        }

        std::cout << "================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 
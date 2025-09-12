#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <omp.h>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

static std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v;
    if (n <= 1) { v.push_back(a); return v; }
    v.resize(n);
    double step = (b - a) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) v[i] = a + step * i;
    return v;
}

// Compute average position MSE across all runs for LINEAR model
static double computeAvgPositionMSELinear(
    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
    const std::vector<double> &dt_vec,
    double dt,
    double V0,
    double meas_noise_std,
    int optimizer_max_iters,
    bool optimizer_verbose,
    const std::string &optimizer_init_mode,
    double optimizer_pos_std,
    double optimizer_vel_std)
{
    const int num_graphs = static_cast<int>(all_states.size());
    const int T = static_cast<int>(all_states[0].size());

    std::vector<double> run_mse(num_graphs, std::numeric_limits<double>::quiet_NaN());

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        try {
            FactorGraph2DTrajectory fg;
            fg.setQFromProcessNoiseIntensity(V0, dt);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);

            fg.setMaxIterations(optimizer_max_iters);
            fg.setVerbose(optimizer_verbose);
            fg.setInitMode(optimizer_init_mode);
            fg.setInitJitter(optimizer_pos_std, optimizer_vel_std);

            if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
                fg.run(all_states[run], &all_measurements[run], dt_vec, true);
            } else {
                fg.run(all_states[run], &all_measurements[run], dt, true);
            }

            auto est_states = fg.getAllEstimates();
            const auto &true_states = all_states[run];

            double sum_sq = 0.0;
            for (int k = 0; k < T; ++k) {
                Eigen::Vector2d e = est_states[k].head<2>() - true_states[k].head<2>();
                sum_sq += e.squaredNorm();
            }
            run_mse[run] = sum_sq / static_cast<double>(T * 2);
        } catch (...) {
            // Leave as NaN if failure
        }
    }

    double total = 0.0; int count = 0;
    for (double v : run_mse) {
        if (std::isfinite(v)) { total += v; count++; }
    }
    if (count == 0) return std::numeric_limits<double>::infinity();
    return total / static_cast<double>(count);
}

int main() {
    std::cout << "=== Grid Search (Average Position MSE) for Linear Tracking ===" << std::endl;
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");

    // Load linear data
    std::string states_file = "../2D-Tracking/Saved_Data/2D_noisy_states.h5";
    std::string measurements_file = "../2D-Tracking/Saved_Data/2D_noisy_measurements.h5";

    std::vector<std::vector<Eigen::Vector4d>> all_states;
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    try {
        all_states = load_all_noisy_states_h5(states_file);
        all_measurements = load_all_noisy_measurements_h5(measurements_file);
        std::cout << "Loaded " << all_states.size() << " linear trajectories" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error loading linear data: " << e.what() << std::endl;
        return -1;
    }

    if (all_states.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return -1;
    }

    const int num_graphs = static_cast<int>(all_states.size());
    const int T = static_cast<int>(all_states[0].size());
    const double dt = config["Data_Generation"]["dt"].as<double>();

    // Build per-step dt schedule
    std::vector<double> dt_vec(std::max(0, T - 1), dt);
    if (config["Data_Generation"]["dt_pieces"]) {
        for (const auto &piece : config["Data_Generation"]["dt_pieces"]) {
            int from = piece["from"].as<int>();
            int to = piece["to"].as<int>();
            double dt_piece = piece["dt"].as<double>();
            from = std::max(0, from);
            to = std::min(T - 2, to);
            for (int k = from; k <= to; ++k) dt_vec[k] = dt_piece;
        }
    }

    // Grid search parameters
    double q_min = config["parameters"][0]["lower_bound"].as<double>();
    double q_max = config["parameters"][0]["upper_bound"].as<double>();
    double R_min = config["parameters"][1]["lower_bound"].as<double>();
    double R_max = config["parameters"][1]["upper_bound"].as<double>();
    int q_points = 50;
    int R_points = 50;
    std::string out_h5 = "../2D-Tracking/Saved_Data/2D_gridsearch_linear_trials_mse.h5";

    if (config["mse_grid_search"]) {
        auto gs = config["mse_grid_search"];
        q_min = gs["q_min"].as<double>(q_min);
        q_max = gs["q_max"].as<double>(q_max);
        R_min = gs["R_min"].as<double>(R_min);
        R_max = gs["R_max"].as<double>(R_max);
        q_points = gs["q_points"].as<int>(q_points);
        R_points = gs["R_points"].as<int>(R_points);
        try {
            std::string suggested = gs["output_filename"].as<std::string>(out_h5);
            if (!suggested.empty()) out_h5 = suggested;
        } catch (...) {}
    } else if (config["grid_search"]) {
        auto gs = config["grid_search"];
        q_min = gs["q_min"].as<double>(q_min);
        q_max = gs["q_max"].as<double>(q_max);
        R_min = gs["R_min"].as<double>(R_min);
        R_max = gs["R_max"].as<double>(R_max);
        q_points = gs["q_points"].as<int>(q_points);
        R_points = gs["R_points"].as<int>(R_points);
        try {
            std::string suggested = gs["output_filename"].as<std::string>(out_h5);
            if (!suggested.empty()) {
                size_t pos = suggested.rfind('.');
                if (pos != std::string::npos) out_h5 = suggested.substr(0, pos) + "_mse" + suggested.substr(pos);
                else out_h5 = suggested + "_mse.h5";
            }
        } catch (...) {}
    }

    // Optimizer settings (we always optimize to get estimates)
    int opt_max_iters = config["optimizer"]["max_iterations"].as<int>(100);
    bool opt_verbose = config["optimizer"]["verbose"].as<bool>(false);
    std::string opt_init_mode = config["optimizer"]["init_mode"].as<std::string>("measurement");
    double opt_pos_std = config["optimizer"]["init_jitter"]["pos_std"].as<double>(0.05);
    double opt_vel_std = config["optimizer"]["init_jitter"]["vel_std"].as<double>(0.2);

    std::cout << "Grid settings: Q in [" << q_min << ", " << q_max << "] (" << q_points << ")"
              << ", R in [" << R_min << ", " << R_max << "] (" << R_points << ")" << std::endl;
    std::cout << "Metric: Average Position MSE" << std::endl;

    std::vector<double> q_vals = linspace(q_min, q_max, q_points);
    std::vector<double> R_vals = linspace(R_min, R_max, R_points);

    std::vector<double> q_out; q_out.reserve(static_cast<size_t>(q_points) * static_cast<size_t>(R_points));
    std::vector<double> r_out; r_out.reserve(q_out.capacity());
    std::vector<double> c_out; c_out.reserve(q_out.capacity());

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iq = 0; iq < q_points; ++iq) {
        for (int ir = 0; ir < R_points; ++ir) {
            double V0 = q_vals[iq];
            double Rvar = R_vals[ir];
            double meas_std = std::sqrt(Rvar);
            auto item_start = std::chrono::high_resolution_clock::now();

            double avg_mse = computeAvgPositionMSELinear(
                all_states,
                all_measurements,
                dt_vec,
                dt,
                V0,
                meas_std,
                opt_max_iters,
                opt_verbose,
                opt_init_mode,
                opt_pos_std,
                opt_vel_std
            );
            auto item_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> item_dur = item_end - item_start;
            std::cout << "  Cell (row " << (iq + 1) << "/" << q_points
                      << ", col " << (ir + 1) << "/" << R_points
                      << ") q=" << V0 << ", R=" << Rvar
                      << " took " << item_dur.count() << " s" << std::endl;

            q_out.push_back(V0);
            r_out.push_back(Rvar);
            c_out.push_back(avg_mse);
        }
        std::cout << "Row " << (iq + 1) << "/" << q_points << " completed." << std::endl;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Grid search (MSE) completed in " << ms << " ms" << std::endl;

    // Real HDF5 writing
    try {
        H5::H5File file(out_h5, H5F_ACC_TRUNC);
        auto write1d = [&](const std::string &name, const std::vector<double> &data) {
            hsize_t dims[1] = { static_cast<hsize_t>(data.size()) };
            H5::DataSpace space(1, dims);
            H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        write1d("q_values", q_out);
        write1d("r_values", r_out);
        write1d("objective_values", c_out);

        // Attributes
        H5::DataSpace scalar(H5S_SCALAR);
        auto writeDoubleAttr = [&](const std::string &name, double v) {
            H5::Attribute a = file.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar);
            a.write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        auto writeIntAttr = [&](const std::string &name, int v) {
            H5::Attribute a = file.createAttribute(name, H5::PredType::NATIVE_INT, scalar);
            a.write(H5::PredType::NATIVE_INT, &v);
        };
        writeIntAttr("num_graphs", num_graphs);
        writeIntAttr("trajectory_length", T);
        writeDoubleAttr("dt_default", dt);
        const std::string metric = "avg_position_mse";
        H5::StrType strType(H5::PredType::C_S1, metric.size());
        H5::Attribute a = file.createAttribute("metric", strType, scalar);
        a.write(strType, metric);

        file.close();
        std::cout << "Saved MSE grid to: " << out_h5 << std::endl;
    } catch (H5::Exception &e) {
        std::cerr << "Error writing HDF5: " << e.getDetailMsg() << std::endl;
        return 1;
    }

    return 0;
} 
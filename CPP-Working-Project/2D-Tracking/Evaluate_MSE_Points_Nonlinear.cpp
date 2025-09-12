#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>

struct QRPoint { double q; double R; };

static std::vector<QRPoint> parsePointsCLI(const std::string& s) {
    // Format: "q1,R1;q2,R2;..."
    std::vector<QRPoint> pts;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ';')) {
        if (token.empty()) continue;
        std::stringstream ts(token);
        std::string qstr, rstr;
        if (std::getline(ts, qstr, ',') && std::getline(ts, rstr)) {
            try {
                double qv = std::stod(qstr);
                double rv = std::stod(rstr);
                pts.push_back({qv, rv});
            } catch (...) { /* skip malformed */ }
        }
    }
    return pts;
}

static std::vector<QRPoint> loadPointsFromYAML(const YAML::Node& config) {
    std::vector<QRPoint> pts;
    // Prefer mse_eval.tests if present
    if (config["mse_eval"] && config["mse_eval"]["tests"]) {
        for (const auto& t : config["mse_eval"]["tests"]) {
            if (t["q"] && t["R"]) {
                pts.push_back({ t["q"].as<double>(), t["R"].as<double>() });
            }
        }
        if (!pts.empty()) return pts;
    }
    // Fallback to nis_eval.tests for convenience
    if (config["nis_eval"] && config["nis_eval"]["tests"]) {
        for (const auto& t : config["nis_eval"]["tests"]) {
            if (t["q"] && t["R"]) {
                pts.push_back({ t["q"].as<double>(), t["R"].as<double>() });
            }
        }
    }
    return pts;
}

// Compute average position MSE across all runs for NONLINEAR model
static double computeAvgPositionMSENonlinear(
    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
    const std::vector<double> &dt_vec,
    double dt,
    double V0,
    double meas_noise_std,
    double turn_rate,
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
            fg.setMotionModelType("constant_turn_rate", turn_rate);
            fg.setMeasurementModelType("gps");
            fg.setQFromProcessNoiseIntensity(V0, dt);
            fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);

            fg.setMaxIterations(optimizer_max_iters);
            fg.setVerbose(optimizer_verbose);
            fg.setInitMode(optimizer_init_mode);
            fg.setInitJitter(optimizer_pos_std, optimizer_vel_std);

            if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
                fg.runNonlinear(all_states[run], &all_measurements[run], dt_vec, true);
            } else {
                fg.runNonlinear(all_states[run], &all_measurements[run], dt, true);
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
            // leave NaN
        }
    }

    double total = 0.0; int count = 0;
    for (double v : run_mse) {
        if (std::isfinite(v)) { total += v; count++; }
    }
    if (count == 0) return std::numeric_limits<double>::infinity();
    return total / static_cast<double>(count);
}

int main(int argc, char* argv[]) {
    std::cout << "=== Evaluate Avg Position MSE at Specific (Q,R) Points (Nonlinear) ===" << std::endl;

    // Parse optional CLI for points (still supported)
    std::string points_arg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--points" || a == "-p") && i + 1 < argc) {
            points_arg = argv[++i];
        }
    }

    // Load scenario
    YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");

    // Nonlinear data files
    std::string states_file = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
    std::string measurements_file = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";

    // Load data
    std::vector<std::vector<Eigen::Vector4d>> all_states = load_all_noisy_states_h5(states_file);
    std::vector<std::vector<Eigen::Vector2d>> all_measurements = load_all_noisy_measurements_h5(measurements_file);
    if (all_states.empty()) { std::cerr << "No trajectories found. Exiting." << std::endl; return 1; }

    const int T = static_cast<int>(all_states[0].size());
    const double dt = config["Data_Generation"]["dt"].as<double>();

    // Build dt_vec
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

    const double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);

    // Optimizer settings
    int opt_max_iters = config["optimizer"]["max_iterations"].as<int>(100);
    bool opt_verbose = config["optimizer"]["verbose"].as<bool>(false);
    std::string opt_init_mode = config["optimizer"]["init_mode"].as<std::string>("measurement");
    double opt_pos_std = config["optimizer"]["init_jitter"]["pos_std"].as<double>(0.05);
    double opt_vel_std = config["optimizer"]["init_jitter"]["vel_std"].as<double>(0.2);

    // Points to evaluate: prefer YAML (mse_eval.tests), fallback to nis_eval.tests, then CLI
    std::vector<QRPoint> points = loadPointsFromYAML(config);
    if (points.empty() && !points_arg.empty()) {
        points = parsePointsCLI(points_arg);
    }
    if (points.empty()) {
        std::cout << "No points provided in YAML (mse_eval.tests/nis_eval.tests) or via --points. Example:" << std::endl;
        std::cout << "  ./Evaluate_MSE_Points_Nonlinear --points 1.0,2.0;1.5,4.0" << std::endl;
        return 1;
    }

    std::cout << "Evaluating " << points.size() << " points..." << std::endl;

    std::vector<double> q_out, r_out, mse_out, time_out;
    q_out.reserve(points.size()); r_out.reserve(points.size());
    mse_out.reserve(points.size()); time_out.reserve(points.size());

    auto start_all = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < points.size(); ++i) {
        double V0 = points[i].q;
        double Rvar = points[i].R;
        double meas_std = std::sqrt(Rvar);
        auto t0 = std::chrono::high_resolution_clock::now();
        double avg_mse = computeAvgPositionMSENonlinear(
            all_states, all_measurements, dt_vec, dt,
            V0, meas_std, turn_rate,
            opt_max_iters, opt_verbose, opt_init_mode, opt_pos_std, opt_vel_std
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt_s = t1 - t0;

        std::cout << "Point " << (i+1) << "/" << points.size()
                  << ": Q=" << V0 << ", R=" << Rvar
                  << ", AvgMSE=" << avg_mse
                  << ", time=" << dt_s.count() << " s" << std::endl;

        q_out.push_back(V0);
        r_out.push_back(Rvar);
        mse_out.push_back(avg_mse);
        time_out.push_back(dt_s.count());
    }
    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur_all = end_all - start_all;
    std::cout << "Completed MSE evaluation for all points in " << dur_all.count() << " s" << std::endl;

    // Save to HDF5 and CSV
    try {
        std::string out_h5 = "../2D-Tracking/Saved_Data/2D_mse_point_evals.h5";
        H5::H5File file(out_h5, H5F_ACC_TRUNC);
        auto write1d = [&](const std::string &name, const std::vector<double> &data) {
            hsize_t dims[1] = { static_cast<hsize_t>(data.size()) };
            H5::DataSpace space(1, dims);
            H5::DataSet ds = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        write1d("q_values", q_out);
        write1d("r_values", r_out);
        write1d("avg_mse", mse_out);
        write1d("eval_seconds", time_out);
        file.close();
        std::cout << "Saved H5: " << out_h5 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "H5 save failed: " << e.what() << std::endl;
    }

    try {
        std::string out_csv = "../2D-Tracking/Saved_Data/2D_mse_point_evals.csv";
        std::ofstream ofs(out_csv);
        ofs << "q,R,avg_mse,seconds\n";
        for (size_t i = 0; i < q_out.size(); ++i) {
            ofs << q_out[i] << "," << r_out[i] << "," << mse_out[i] << "," << time_out[i] << "\n";
        }
        ofs.close();
        std::cout << "Saved CSV: " << out_csv << std::endl;
    } catch (...) {}

    return 0;
}

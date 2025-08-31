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

// Helper: infer num_graphs and trajectory length from HDF5
void infer_problem_size_h5(const std::string &filename, int &num_graphs, int &Trajectory_length) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("states");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims, nullptr);
        num_graphs = static_cast<int>(dims[0]);
        Trajectory_length = static_cast<int>(dims[1]);
    } catch (H5::Exception &e) {
        std::cerr << "Error inferring problem size from HDF5: " << e.getDetailMsg() << std::endl;
        num_graphs = 0;
        Trajectory_length = 0;
    }
}

// Compute CNEES over all runs using full information matrix
static double computeCNEES_linear(
    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
    const std::vector<double> &dt_vec,
    double dt,
    double V0,
    double meas_noise_std)
{
    const int num_graphs = static_cast<int>(all_states.size());
    const int T = static_cast<int>(all_states[0].size());
    const int nx = 4;

    std::vector<double> nees_full_system(num_graphs, 0.0);

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        fg.setQFromProcessNoiseIntensity(V0, dt);
        fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
        FactorGraph2DTrajectory::OutputOptions opts;
        opts.output_estimated_state = true;
        opts.output_true_state = true;
        opts.output_information_matrix = true;
        fg.setOutputOptions(opts);
        if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
            fg.run(all_states[run], &all_measurements[run], dt_vec, true);
        } else {
            fg.run(all_states[run], &all_measurements[run], dt);
        }
        auto est_states = fg.getAllEstimates();
        auto true_states = fg.getAllTrueStates();
        Eigen::MatrixXd Hessian = fg.getFullHessianMatrix();
        if (Hessian.rows() == 0 || Hessian.cols() == 0) {
            continue;
        }
        Eigen::VectorXd full_error(T * nx);
        for (int k = 0; k < T; ++k) {
            Eigen::Vector4d err = true_states[k] - est_states[k];
            full_error.segment<4>(k * nx) = err;
        }
        double nees_full = full_error.transpose() * Hessian * full_error;
        nees_full_system[run] = nees_full;
    }

    double mean_nees = 0.0;
    for (int run = 0; run < num_graphs; ++run) mean_nees += nees_full_system[run];
    mean_nees /= num_graphs;

    double variance_nees = 0.0;
    for (int run = 0; run < num_graphs; ++run) {
        double diff = nees_full_system[run] - mean_nees;
        variance_nees += diff * diff;
    }
    if (num_graphs > 1) variance_nees /= (num_graphs - 1);

    // DOF from actual graph
    FactorGraph2DTrajectory temp_fg;
    temp_fg.setQFromProcessNoiseIntensity(V0, dt);
    temp_fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
    if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
        temp_fg.run(all_states[0], &all_measurements[0], dt_vec, true);
    } else {
        temp_fg.run(all_states[0], &all_measurements[0], dt);
    }
    auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();
    int total_dof = dimX;

    double log_mean = std::log(mean_nees / total_dof);
    double log_variance = (variance_nees > 0) ? std::log(variance_nees / (2.0 * total_dof)) : 0.0;
    return std::abs(log_mean) + std::abs(log_variance);
}

// Compute CNIS over all runs (Proposition 3 or 4)
static double computeCNIS_linear(
    const std::vector<std::vector<Eigen::Vector4d>> &all_states,
    const std::vector<std::vector<Eigen::Vector2d>> &all_measurements,
    const std::vector<double> &dt_vec,
    double dt,
    double V0,
    double meas_noise_std,
    const std::string &consistency_method,
    int optimizer_max_iters,
    bool optimizer_verbose,
    const std::string &optimizer_init_mode,
    double optimizer_pos_std,
    double optimizer_vel_std)
{
    const int num_graphs = static_cast<int>(all_states.size());
    const int T = static_cast<int>(all_states[0].size());
    const int nx = 4;
    const int nz = 2;

    std::vector<double> nis_full_system(num_graphs, 0.0);

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        fg.setQFromProcessNoiseIntensity(V0, dt);
        fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);

        bool do_optimization = (consistency_method == "nis4");
        if (do_optimization) {
            fg.setMaxIterations(optimizer_max_iters);
            fg.setVerbose(optimizer_verbose);
            fg.setInitMode(optimizer_init_mode);
            fg.setInitJitter(optimizer_pos_std, optimizer_vel_std);
        }

        if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
            fg.run(all_states[run], &all_measurements[run], dt_vec, do_optimization);
        } else {
            fg.run(all_states[run], &all_measurements[run], dt, do_optimization);
        }
        nis_full_system[run] = fg.getChi2();
    }

    double mean_nis = 0.0;
    for (int run = 0; run < num_graphs; ++run) mean_nis += nis_full_system[run];
    mean_nis /= num_graphs;

    double variance_nis = 0.0;
    for (int run = 0; run < num_graphs; ++run) {
        double diff = nis_full_system[run] - mean_nis;
        variance_nis += diff * diff;
    }
    if (num_graphs > 1) variance_nis /= (num_graphs - 1);

    FactorGraph2DTrajectory temp_fg;
    temp_fg.setQFromProcessNoiseIntensity(V0, dt);
    temp_fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
    if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
        temp_fg.run(all_states[0], &all_measurements[0], dt_vec, true);
    } else {
        temp_fg.run(all_states[0], &all_measurements[0], dt);
    }
    auto [dimZ, dimX] = temp_fg.getActualGraphDimensions();

    int total_dof;
    if (consistency_method == "nis3") {
        total_dof = dimZ;
    } else if (consistency_method == "nis4") {
        total_dof = dimZ - dimX;
    } else {
        total_dof = nz;
    }

    double log_mean = std::log(mean_nis / total_dof);
    double log_variance = (variance_nis > 0) ? std::log(variance_nis / (2.0 * total_dof)) : 0.0;
    return std::abs(log_mean) + std::abs(log_variance);
}

static std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v;
    if (n <= 1) { v.push_back(a); return v; }
    v.resize(n);
    double step = (b - a) / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) v[i] = a + step * i;
    return v;
}

int main() {
    YAML::Node config = YAML::LoadFile("../scenario_linear.yaml");

    // dt and optional schedule
    double dt = config["Data_Generation"]["dt"].as<double>();
    int Trajectory_length = 0; int num_graphs = 0;
    std::vector<double> dt_vec;

    infer_problem_size_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from HDF5." << std::endl;
        return 1;
    }
    dt_vec.assign(std::max(0, Trajectory_length - 1), dt);
    if (config["Data_Generation"]["dt_pieces"]) {
        for (const auto &piece : config["Data_Generation"]["dt_pieces"]) {
            int from = piece["from"].as<int>();
            int to = piece["to"].as<int>();
            double dt_piece = piece["dt"].as<double>();
            from = std::max(0, from);
            to = std::min(Trajectory_length - 2, to);
            for (int k = from; k <= to; ++k) dt_vec[k] = dt_piece;
        }
    }

    auto all_states = load_all_noisy_states_h5("../2D-Tracking/Saved_Data/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../2D-Tracking/Saved_Data/2D_noisy_measurements.h5");

    // Grid search parameters
    double q_min = config["parameters"][0]["lower_bound"].as<double>();
    double q_max = config["parameters"][0]["upper_bound"].as<double>();
    double R_min = config["parameters"][1]["lower_bound"].as<double>();
    double R_max = config["parameters"][1]["upper_bound"].as<double>();
    int q_points = 50;
    int R_points = 50;
    std::string consistency_method = config["bayesopt"]["consistency_method"].as<std::string>("nis3");
    bool save_h5 = true;
    std::string out_h5 = "../2D-Tracking/Saved_Data/2D_gridsearch_trials.h5";

    if (config["grid_search"]) {
        auto gs = config["grid_search"];
        q_min = gs["q_min"].as<double>(q_min);
        q_max = gs["q_max"].as<double>(q_max);
        R_min = gs["R_min"].as<double>(R_min);
        R_max = gs["R_max"].as<double>(R_max);
        q_points = gs["q_points"].as<int>(q_points);
        R_points = gs["R_points"].as<int>(R_points);
        consistency_method = gs["consistency_method"].as<std::string>(consistency_method);
        save_h5 = gs["save_h5"].as<bool>(save_h5);
        out_h5 = gs["output_filename"].as<std::string>(out_h5);
    }

    // Optimizer settings used only for nis4
    int opt_max_iters = config["optimizer"]["max_iterations"].as<int>(100);
    bool opt_verbose = config["optimizer"]["verbose"].as<bool>(false);
    std::string opt_init_mode = config["optimizer"]["init_mode"].as<std::string>("measurement");
    double opt_pos_std = config["optimizer"]["init_jitter"]["pos_std"].as<double>(0.05);
    double opt_vel_std = config["optimizer"]["init_jitter"]["vel_std"].as<double>(0.2);

    std::cout << "=== Linear Grid Search Tracking Test ===" << std::endl;
    std::cout << "Grid q in [" << q_min << ", " << q_max << "] with " << q_points << " points\n";
    std::cout << "Grid R in [" << R_min << ", " << R_max << "] with " << R_points << " points\n";
    std::cout << "Consistency method: " << consistency_method << std::endl;

    std::vector<double> q_vals = linspace(q_min, q_max, q_points);
    std::vector<double> R_vals = linspace(R_min, R_max, R_points);

    std::vector<std::array<double, 3>> trials;
    trials.reserve(static_cast<size_t>(q_points) * static_cast<size_t>(R_points));

    double best_obj = std::numeric_limits<double>::infinity();
    double best_q = q_vals.front();
    double best_R = R_vals.front();

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int iq = 0; iq < q_points; ++iq) {
        for (int iR = 0; iR < R_points; ++iR) {
            double q = q_vals[iq];
            double R = R_vals[iR];
            double meas_std = std::sqrt(R);
            double metric = 0.0;
            if (consistency_method == "nis3" || consistency_method == "nis4") {
                metric = computeCNIS_linear(all_states, all_measurements, dt_vec, dt, q, meas_std,
                                            consistency_method, opt_max_iters, opt_verbose, opt_init_mode, opt_pos_std, opt_vel_std);
            } else {
                metric = computeCNEES_linear(all_states, all_measurements, dt_vec, dt, q, meas_std);
            }
            trials.push_back({q, R, metric});
            if (metric < best_obj) { best_obj = metric; best_q = q; best_R = R; }
        }
        std::cout << "Completed row " << (iq + 1) << "/" << q_points << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_dur = total_end - total_start;
    std::cout << "Grid search completed in " << total_dur.count() << " seconds." << std::endl;

    // Save trials to HDF5 (same schema as BO linear: Nx3 dataset named 'trials')
    if (save_h5) {
        try {
            hsize_t dims[2] = {trials.size(), 3};
            H5::H5File file(out_h5, H5F_ACC_TRUNC);
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("trials", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> flat;
            flat.reserve(trials.size() * 3);
            for (const auto &row : trials) flat.insert(flat.end(), row.begin(), row.end());
            dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
            file.close();
            std::cout << "Saved trials to: " << out_h5 << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Failed to save HDF5 trials: " << e.what() << std::endl;
        }
    }

    // Update validate_filter with best
    try {
        YAML::Node outc = YAML::LoadFile("../scenario_linear.yaml");
        outc["validate_filter"]["q"] = best_q;
        outc["validate_filter"]["R"] = best_R;
        outc["validate_filter"]["min_objective"] = best_obj;
        std::ofstream yaml_out("../scenario_linear.yaml");
        yaml_out << outc;
        yaml_out.close();
    } catch (const std::exception &e) {
        std::cerr << "Failed to update validate_filter: " << e.what() << std::endl;
    }

    // Save summary
    try {
        std::ofstream txt("../2D-Tracking/Saved_Data/2D_gridsearch_best_linear.txt");
        txt << "Best q: " << best_q << "\n";
        txt << "Best R: " << best_R << "\n";
        txt << "Best objective: " << best_obj << "\n";
        txt << "q_points: " << q_points << ", R_points: " << R_points << "\n";
        txt.close();
    } catch (...) {}

    std::cout << "Best q=" << best_q << ", R=" << best_R << ", objective=" << best_obj << std::endl;
    return 0;
} 
#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

struct TestPair { double q; double R; };

static void build_dt_vec_from_yaml(const YAML::Node &config, int T, std::vector<double> &dt_vec) {
    double dt = config["Data_Generation"]["dt"].as<double>();
    dt_vec.assign(std::max(0, T - 1), dt);
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
}

int main() {
    YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");

    std::string states_file = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
    std::string meas_file = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";

    auto all_states = load_all_noisy_states_h5(states_file);
    auto all_measurements = load_all_noisy_measurements_h5(meas_file);

    int num_graphs = static_cast<int>(all_states.size());
    if (num_graphs == 0) {
        std::cerr << "No trajectories found." << std::endl;
        return 1;
    }
    int T = static_cast<int>(all_states[0].size());

    std::vector<double> dt_vec; build_dt_vec_from_yaml(config, T, dt_vec);
    double dt = config["Data_Generation"]["dt"].as<double>();
    double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);

    // Read tests from YAML
    std::vector<TestPair> tests;
    std::string out_h5 = "../2D-Tracking/Saved_Data/2D_nis_nonlinear.h5";
    std::string method = config["bayesopt"]["consistency_method"].as<std::string>("nis3");
    if (config["nis_eval"]) {
        const auto &ne = config["nis_eval"];
        method = ne["method"].as<std::string>(method);
        out_h5 = ne["output"].as<std::string>(out_h5);
        if (ne["tests"]) {
            for (const auto &t : ne["tests"]) {
                tests.push_back({t["q"].as<double>(), t["R"].as<double>()});
            }
        }
    }
    if (tests.empty()) {
        tests.push_back({config["validate_filter"]["q"].as<double>(), config["validate_filter"]["R"].as<double>()});
    }

    std::vector<double> chi2_values;
    chi2_values.resize(static_cast<size_t>(tests.size()) * static_cast<size_t>(num_graphs), 0.0);

    for (size_t ti = 0; ti < tests.size(); ++ti) {
        double q = tests[ti].q;
        double Rvar = tests[ti].R;
        double Rstd = std::sqrt(Rvar);

        #pragma omp parallel for
        for (int run = 0; run < num_graphs; ++run) {
            FactorGraph2DTrajectory fg;
            fg.setMotionModelType("constant_turn_rate", turn_rate);
            fg.setMeasurementModelType("gps");
            fg.setQFromProcessNoiseIntensity(q, dt);
            fg.setRFromMeasurementNoise(Rstd, Rstd);

            bool do_optimization = (method == "nis4");
            if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
                fg.runNonlinear(all_states[run], &all_measurements[run], dt_vec, do_optimization);
            } else {
                fg.runNonlinear(all_states[run], &all_measurements[run], dt, do_optimization);
            }
            double chi2 = fg.getChi2();
            chi2_values[ti * static_cast<size_t>(num_graphs) + static_cast<size_t>(run)] = chi2;
        }
    }

    // DOF per test
    std::vector<double> dof_values(tests.size(), 0.0);
    for (size_t ti = 0; ti < tests.size(); ++ti) {
        FactorGraph2DTrajectory tmp;
        tmp.setMotionModelType("constant_turn_rate", turn_rate);
        tmp.setMeasurementModelType("gps");
        tmp.setQFromProcessNoiseIntensity(tests[ti].q, dt);
        tmp.setRFromMeasurementNoise(std::sqrt(tests[ti].R), std::sqrt(tests[ti].R));
        if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
            tmp.runNonlinear(all_states[0], &all_measurements[0], dt_vec, (method == "nis4"));
        } else {
            tmp.runNonlinear(all_states[0], &all_measurements[0], dt, (method == "nis4"));
        }
        auto [dimZ, dimX] = tmp.getActualGraphDimensions();
        dof_values[ti] = (method == "nis4") ? (dimZ - dimX) : dimZ;
    }

    // Compute and print CNIS per test
    std::cout << "CNIS summary (" << method << "):\n";
    for (size_t ti = 0; ti < tests.size(); ++ti) {
        const size_t offset = ti * static_cast<size_t>(num_graphs);
        double mean = 0.0;
        for (int run = 0; run < num_graphs; ++run) mean += chi2_values[offset + static_cast<size_t>(run)];
        mean /= static_cast<double>(num_graphs);
        double var = 0.0;
        for (int run = 0; run < num_graphs; ++run) {
            double d = chi2_values[offset + static_cast<size_t>(run)] - mean;
            var += d * d;
        }
        if (num_graphs > 1) var /= static_cast<double>(num_graphs - 1);
        double dof = dof_values[ti] > 0.0 ? dof_values[ti] : 1.0;
        double log_mean = std::log(std::max(1e-12, mean / dof));
        double log_var_term = (num_graphs > 1) ? (var / (2.0 * dof)) : 0.0;
        double log_variance = (var > 0.0) ? std::log(std::max(1e-12, log_var_term)) : 0.0;
        double cnis = std::abs(log_mean) + std::abs(log_variance);
        std::cout << "  test " << (ti + 1) << ": q=" << tests[ti].q
                  << ", R=" << tests[ti].R
                  << ", DOF=" << dof
                  << ", mean(chi2)=" << mean
                  << ", var(chi2)=" << var
                  << ", CNIS=" << cnis << std::endl;
    }

    // Save HDF5
    try {
        H5::H5File file(out_h5, H5F_ACC_TRUNC);
        hsize_t test_dim[1] = {static_cast<hsize_t>(tests.size())};
        hsize_t runs_dim[1] = {static_cast<hsize_t>(num_graphs)};
        hsize_t chi2_dims[2] = {static_cast<hsize_t>(tests.size()), static_cast<hsize_t>(num_graphs)};

        H5::DataSpace test_space(1, test_dim);
        H5::DataSet qds = file.createDataSet("q_values", H5::PredType::NATIVE_DOUBLE, test_space);
        H5::DataSet rds = file.createDataSet("r_values", H5::PredType::NATIVE_DOUBLE, test_space);
        std::vector<double> qvals(tests.size()), rvals(tests.size());
        for (size_t i = 0; i < tests.size(); ++i) { qvals[i] = tests[i].q; rvals[i] = tests[i].R; }
        qds.write(qvals.data(), H5::PredType::NATIVE_DOUBLE);
        rds.write(rvals.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet dds = file.createDataSet("dof_values", H5::PredType::NATIVE_DOUBLE, test_space);
        dds.write(dof_values.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSpace chi2_space(2, chi2_dims);
        H5::DataSet cds = file.createDataSet("chi2_values", H5::PredType::NATIVE_DOUBLE, chi2_space);
        cds.write(chi2_values.data(), H5::PredType::NATIVE_DOUBLE);

        file.close();
        std::cout << "Saved NIS (chi2) to: " << out_h5 << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Failed to save HDF5: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 
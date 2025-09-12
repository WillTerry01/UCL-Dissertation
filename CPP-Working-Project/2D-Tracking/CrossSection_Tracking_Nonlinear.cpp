#include "fg_class_tracking.h"
#include "2D_h5_loader.h"
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

static double compute_cnis_for_candidate(
    const std::vector<std::vector<Eigen::Vector4d>>& all_states,
    const std::vector<std::vector<Eigen::Vector2d>>& all_measurements,
    const std::vector<double>& dt_vec,
    double dt,
    double q_intensity,
    double meas_noise_var,
    const std::string& method,
    double turn_rate,
    double* out_term_mean_abs = nullptr,
    double* out_term_var_abs = nullptr)
{
    const int num_graphs = static_cast<int>(all_states.size());
    const int T = static_cast<int>(all_states[0].size());
    const int nx = 4;
    const int nz = 2;

    const bool do_optimization = (method == "nis4");

    std::vector<double> nis_values(num_graphs, 0.0);

    #pragma omp parallel for
    for (int run = 0; run < num_graphs; ++run) {
        FactorGraph2DTrajectory fg;
        fg.setMotionModelType("constant_turn_rate", turn_rate);
        fg.setMeasurementModelType("gps");
        fg.setQFromProcessNoiseIntensity(q_intensity, dt);
        const double meas_noise_std = std::sqrt(meas_noise_var);
        fg.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);

        if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
            fg.runNonlinear(all_states[run], &all_measurements[run], dt_vec, do_optimization);
        } else {
            fg.runNonlinear(all_states[run], &all_measurements[run], dt, do_optimization);
        }
        double chi2 = fg.getChi2();
        nis_values[run] = chi2;
    }

    // Compute mean and variance
    double mean_nis = 0.0;
    for (int i = 0; i < num_graphs; ++i) mean_nis += nis_values[i];
    mean_nis /= std::max(1, num_graphs);

    double var_nis = 0.0;
    for (int i = 0; i < num_graphs; ++i) {
        const double d = nis_values[i] - mean_nis;
        var_nis += d * d;
    }
    if (num_graphs > 1) var_nis /= (num_graphs - 1);

    // DOF from actual graph
    FactorGraph2DTrajectory tmp;
    tmp.setMotionModelType("constant_turn_rate", turn_rate);
    tmp.setMeasurementModelType("gps");
    tmp.setQFromProcessNoiseIntensity(q_intensity, dt);
    const double meas_noise_std = std::sqrt(meas_noise_var);
    tmp.setRFromMeasurementNoise(meas_noise_std, meas_noise_std);
    if (!dt_vec.empty() && static_cast<int>(dt_vec.size()) == T - 1) {
        tmp.runNonlinear(all_states[0], &all_measurements[0], dt_vec, do_optimization);
    } else {
        tmp.runNonlinear(all_states[0], &all_measurements[0], dt, do_optimization);
    }
    auto dims = tmp.getActualGraphDimensions();
    const int dimZ = dims.first;
    const int dimX = dims.second;
    const int dof = (method == "nis4") ? (dimZ - dimX) : dimZ;

    // CNIS = |log(mean/DOF)| + |log(var/(2*DOF))|
    const double term_mean = std::log(mean_nis / std::max(1, dof));
    const double term_var = (var_nis > 0.0) ? std::log(var_nis / (2.0 * std::max(1, dof))) : 0.0;
    const double term_mean_abs = std::abs(term_mean);
    const double term_var_abs = std::abs(term_var);
    if (out_term_mean_abs) *out_term_mean_abs = term_mean_abs;
    if (out_term_var_abs) *out_term_var_abs = term_var_abs;
    const double CNIS = term_mean_abs + term_var_abs;
    return CNIS;
}

int main(int argc, char* argv[]) {
    try {
        YAML::Node config = YAML::LoadFile("../scenario_nonlinear.yaml");
        const std::string method = (argc > 1) ? std::string(argv[1]) : config["bayesopt"]["consistency_method"].as<std::string>("nis4");
        const int num_points = (argc > 2) ? std::max(10, std::stoi(argv[2])) : 200;

        // Load data
        const std::string states_file = "../2D-Tracking/Saved_Data/2D_nonlinear_states.h5";
        const std::string measurements_file = "../2D-Tracking/Saved_Data/2D_nonlinear_measurements.h5";
        auto all_states = load_all_noisy_states_h5(states_file);
        auto all_measurements = load_all_noisy_measurements_h5(measurements_file);
        if (all_states.empty()) {
            std::cerr << "No states loaded. Generate data first." << std::endl;
            return 1;
        }
        const int T = static_cast<int>(all_states[0].size());

        // Params
        const double dt = config["Data_Generation"]["dt"].as<double>();
        const double q_true = config["Data_Generation"]["q"].as<double>();
        const double R_true = config["Data_Generation"]["meas_noise_var"].as<double>();
        const double turn_rate = config["Data_Generation"]["turn_rate"].as<double>(0.1);

        // dt schedule if present
        std::vector<double> dt_vec(std::max(0, T - 1), dt);
        if (config["Data_Generation"]["dt_pieces"]) {
            for (const auto& piece : config["Data_Generation"]["dt_pieces"]) {
                int from = piece["from"].as<int>();
                int to = piece["to"].as<int>();
                double dt_piece = piece["dt"].as<double>();
                from = std::max(0, from);
                to = std::min(T - 2, to);
                for (int k = from; k <= to; ++k) dt_vec[k] = dt_piece;
            }
        }

        // R sweep when locking Q
        const double r_min = 1.95;
        const double r_max = 2.05;
        std::vector<double> r_values(num_points);
        std::vector<double> c_lockQ(num_points);
        std::vector<double> c_lockQ_mean(num_points);
        std::vector<double> c_lockQ_var(num_points);
        for (int i = 0; i < num_points; ++i) {
            r_values[i] = r_min + (r_max - r_min) * (static_cast<double>(i) / (num_points - 1));
            double tmean = 0.0, tvar = 0.0;
            c_lockQ[i] = compute_cnis_for_candidate(all_states, all_measurements, dt_vec, dt, q_true, r_values[i], method, turn_rate, &tmean, &tvar);
            c_lockQ_mean[i] = tmean;
            c_lockQ_var[i] = tvar;
            std::cout << "[lock Q] i=" << i << "/" << num_points << " R=" << r_values[i] << " C=" << c_lockQ[i] << std::endl;
        }

        // Q sweep when locking R
        const double q_min = 0.97;
        const double q_max = 1.03;
        std::vector<double> q_values(num_points);
        std::vector<double> c_lockR(num_points);
        std::vector<double> c_lockR_mean(num_points);
        std::vector<double> c_lockR_var(num_points);
        for (int i = 0; i < num_points; ++i) {
            q_values[i] = q_min + (q_max - q_min) * (static_cast<double>(i) / (num_points - 1));
            double tmean = 0.0, tvar = 0.0;
            c_lockR[i] = compute_cnis_for_candidate(all_states, all_measurements, dt_vec, dt, q_values[i], R_true, method, turn_rate, &tmean, &tvar);
            c_lockR_mean[i] = tmean;
            c_lockR_var[i] = tvar;
            if ((i % 50) == 0) std::cout << "[lock R] i=" << i << "/" << num_points << " Q=" << q_values[i] << " C=" << c_lockR[i] << std::endl;
        }

        // Save to HDF5 (method in filename)
        const std::string h5_path = "../2D-Tracking/Saved_Data/2D_cross_sections_nonlinear_" + method + ".h5";
        H5::H5File file(h5_path, H5F_ACC_TRUNC);

        // lock Q datasets
        {
            hsize_t dims[1] = {static_cast<hsize_t>(num_points)};
            H5::DataSpace space(1, dims);
            H5::DataSet dR = file.createDataSet("lockQ_r_values", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dC = file.createDataSet("lockQ_c_values", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dCm = file.createDataSet("lockQ_c_mean", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dCv = file.createDataSet("lockQ_c_var", H5::PredType::NATIVE_DOUBLE, space);
            dR.write(r_values.data(), H5::PredType::NATIVE_DOUBLE);
            dC.write(c_lockQ.data(), H5::PredType::NATIVE_DOUBLE);
            dCm.write(c_lockQ_mean.data(), H5::PredType::NATIVE_DOUBLE);
            dCv.write(c_lockQ_var.data(), H5::PredType::NATIVE_DOUBLE);
        }
        // lock R datasets
        {
            hsize_t dims[1] = {static_cast<hsize_t>(num_points)};
            H5::DataSpace space(1, dims);
            H5::DataSet dQ = file.createDataSet("lockR_q_values", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dC = file.createDataSet("lockR_c_values", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dCm = file.createDataSet("lockR_c_mean", H5::PredType::NATIVE_DOUBLE, space);
            H5::DataSet dCv = file.createDataSet("lockR_c_var", H5::PredType::NATIVE_DOUBLE, space);
            dQ.write(q_values.data(), H5::PredType::NATIVE_DOUBLE);
            dC.write(c_lockR.data(), H5::PredType::NATIVE_DOUBLE);
            dCm.write(c_lockR_mean.data(), H5::PredType::NATIVE_DOUBLE);
            dCv.write(c_lockR_var.data(), H5::PredType::NATIVE_DOUBLE);
        }
        file.close();

        std::cout << "Saved cross-sections to: " << h5_path << std::endl;
        std::cout << "Method: " << method << ", Points: " << num_points << std::endl;
        std::cout << "Q locked at: " << q_true << ", R sweep: [" << r_min << ", " << r_max << "]" << std::endl;
        std::cout << "R locked at: " << R_true << ", Q sweep: [" << q_min << ", " << q_max << "]" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 
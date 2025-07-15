#include "2D_factor_graph_trajectory.h"
#include "2D_h5_loader.h"
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include "H5Cpp.h"

// Helper function to infer num_graphs and N from the HDF5 file
template <typename T>
void infer_problem_size_h5(const std::string& filename, const std::string& dataset_name, int& num_graphs, int& N) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[3];
        int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
        num_graphs = dims[0];
        N = dims[1];
    } catch (H5::Exception& e) {
        std::cerr << "Error inferring problem size from HDF5: " << e.getDetailMsg() << std::endl;
        num_graphs = 0;
        N = 0;
    }
}

int main() {
    // Infer problem size from the noisy states HDF5
    int Trajectory_length = 0; // Trajectory length
    int num_graphs = 0; // Number of Monte Carlo samples
    infer_problem_size_h5<double>("../H5_Files/2D_noisy_states.h5", "states", num_graphs, Trajectory_length);
    if (Trajectory_length == 0 || num_graphs == 0) {
        std::cerr << "Could not infer problem size from HDF5." << std::endl;
        return 1;
    }
    std::cout << "Detected " << num_graphs << " runs, each of length " << Trajectory_length << std::endl;

    // Load all runs' data from HDF5
    auto all_states = load_all_noisy_states_h5("../H5_Files/2D_noisy_states.h5");
    auto all_measurements = load_all_noisy_measurements_h5("../H5_Files/2D_noisy_measurements.h5");

    if (all_states.size() != num_graphs || all_measurements.size() != num_graphs) {
        std::cerr << "Error: Number of runs in HDF5 does not match detected num_graphs." << std::endl;
        return 1;
    }

    std::vector<double> chi2_values(num_graphs);
    std::vector<std::vector<double>> all_traj_data(num_graphs);
    for (int run = 0; run < num_graphs; ++run) {
        std::cout << "Run " << run << " / " << num_graphs << std::endl;
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * 0.5; // Tune as desired
        fg.R_ = Eigen::Matrix2d::Identity() * 0.5; // Tune as desired
        fg.run(all_states[run], &all_measurements[run], false);
        chi2_values[run] = fg.getChi2();
        all_traj_data[run].reserve(Trajectory_length * 7);
        for (int t = 0; t < Trajectory_length; ++t) {
            const auto& true_state = all_states[run][t];
            const auto& meas = all_measurements[run][t];
            const auto& v = fg.getEstimate(t);
            all_traj_data[run].insert(all_traj_data[run].end(), {
                true_state[0], true_state[1],
                meas[0], meas[1],
                v[0], v[1], 0.0 // pad with 0.0 for est_y if not available
            });
        }
    }
    // Save chi2 values to HDF5
    const std::string chi2_h5 = "../H5_Files/2D_chi2_results.h5";
    hsize_t chi2_dims[1] = {static_cast<hsize_t>(num_graphs)};
    H5::H5File chi2_file(chi2_h5, H5F_ACC_TRUNC);
    H5::DataSpace chi2_space(1, chi2_dims);
    H5::DataSet chi2_dataset = chi2_file.createDataSet("chi2", H5::PredType::NATIVE_DOUBLE, chi2_space);
    chi2_dataset.write(chi2_values.data(), H5::PredType::NATIVE_DOUBLE);
    // Save all trajectory estimates to HDF5
    const std::string traj_h5 = "../H5_Files/2D_trajectory_estimates_all.h5";
    hsize_t traj_dims[3] = {static_cast<hsize_t>(num_graphs), static_cast<hsize_t>(Trajectory_length), 7};
    H5::H5File traj_file(traj_h5, H5F_ACC_TRUNC);
    H5::DataSpace traj_space(3, traj_dims);
    H5::DataSet traj_dataset = traj_file.createDataSet("trajectories", H5::PredType::NATIVE_DOUBLE, traj_space);
    std::vector<double> flat_traj;
    for (const auto& run_data : all_traj_data) flat_traj.insert(flat_traj.end(), run_data.begin(), run_data.end());
    traj_dataset.write(flat_traj.data(), H5::PredType::NATIVE_DOUBLE);
    std::cout << "Saved chi2 values for " << num_graphs << " runs to " << chi2_h5 << std::endl;
    std::cout << "Saved all trajectory estimates to " << traj_h5 << std::endl;

    // Compute the CNEES/CNIS consistency metric
    double meanChi2 = 0.0;
    double covChi2 = 0.0;
    int N = 0;
    // Compute meanChi2
    for (int i = 0; i < num_graphs; ++i) meanChi2 += chi2_values[i];
    meanChi2 /= num_graphs;
    // Compute covChi2
    for (int i = 0; i < num_graphs; ++i) covChi2 += (chi2_values[i] - meanChi2) * (chi2_values[i] - meanChi2);
    covChi2 /= (num_graphs - 1);
    // Degrees of freedom: number of residuals per run (from the first run)
    N = static_cast<int>(all_measurements[0].size()) * 2 + (static_cast<int>(all_states[0].size()) - 1) * 4;
    double C = std::abs(std::log(meanChi2 / N)) + std::abs(std::log(covChi2 / (2.0 * N)));
    std::cout << "Consistency metric C: " << C << std::endl;
    std::ofstream cfile("../H5_Files/2D_consistency_metric.txt");
    cfile << "C = " << C << std::endl;
    cfile.close();
    std::cout << "Saved consistency metric to ../H5_Files/2D_consistency_metric.txt" << std::endl;
    return 0;
} 
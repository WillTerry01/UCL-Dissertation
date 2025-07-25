#include "2D_factor_graph_trajectory.h"
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "H5Cpp.h"

int main() {
    int N = 20;
    int num_runs = 1000;
    std::vector<double> chi2_values(num_runs);

    //calculate the trajectory to run all runs off
    std::vector<Eigen::Vector4d> true_states(N);
    Eigen::Vector2d pos(0.0, 0.0);
    Eigen::Vector2d vel(1.0, 1.0);
    double dt = 0.1;  // Use the same dt as data generation
    true_states[0] << pos.x(), pos.y(), vel.x(), vel.y();
    for (int k = 1; k < N; ++k) {
        true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
        true_states[k].tail<2>() = true_states[k-1].tail<2>();
        
    }
    /*
    std::cout << "true_states:" << std::endl;
    for (int k = 0; k < N; ++k) {
        std::cout << "  [" << k << "]: " << true_states[k].transpose() << std::endl;
    }
    */

    for (int run = 0; run < num_runs; ++run) {
        if (run % 100 == 0) {
            std::cout << "Run " << run << " / " << num_runs << std::endl;
        }
        FactorGraph2DTrajectory fg;
        fg.Q_ = Eigen::Matrix4d::Identity() * 0.110528;
        fg.R_ = Eigen::Matrix2d::Identity() * 0.134589;
        fg.run(true_states, nullptr, true, dt); // Add process noise inside the function
        chi2_values[run] = fg.getChi2();
    }

    // Write chi2 values to HDF5
    const std::string h5_filename = "../H5_Files/2D_chi2_results.h5";
    hsize_t dims[1] = {static_cast<hsize_t>(num_runs)};
    H5::H5File file(h5_filename, H5F_ACC_TRUNC);
    H5::DataSpace dataspace(1, dims);
    H5::DataSet dataset = file.createDataSet("chi2", H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(chi2_values.data(), H5::PredType::NATIVE_DOUBLE);
    std::cout << "Saved chi2 values for " << num_runs << " runs to " << h5_filename << std::endl;
    return 0;
}

#pragma once
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <iostream>
#include "H5Cpp.h"

// Loads all noisy states from HDF5 (dataset: 'states', shape: [num_graphs, N, 4])
inline std::vector<std::vector<Eigen::Vector4d>> load_all_noisy_states_h5(const std::string& filename) {
    std::vector<std::vector<Eigen::Vector4d>> all_states;
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("states");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims, nullptr);
        size_t num_graphs = dims[0];
        size_t N = dims[1];
        std::vector<double> buffer(num_graphs * N * 4);
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        all_states.resize(num_graphs);
        for (size_t run = 0; run < num_graphs; ++run) {
            all_states[run].resize(N);
            for (size_t k = 0; k < N; ++k) {
                size_t idx = run * N * 4 + k * 4;
                Eigen::Vector4d state;
                state << buffer[idx + 0], buffer[idx + 1], buffer[idx + 2], buffer[idx + 3];
                all_states[run][k] = state;
            }
        }
    } catch (H5::Exception& e) {
        std::cerr << "Error loading states from HDF5: " << e.getDetailMsg() << std::endl;
    }
    return all_states;
}

// Loads all noisy measurements from HDF5 (dataset: 'measurements', shape: [num_graphs, N, 2])
inline std::vector<std::vector<Eigen::Vector2d>> load_all_noisy_measurements_h5(const std::string& filename) {
    std::vector<std::vector<Eigen::Vector2d>> all_measurements;
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("measurements");
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims, nullptr);
        size_t num_graphs = dims[0];
        size_t N = dims[1];
        std::vector<double> buffer(num_graphs * N * 2);
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        all_measurements.resize(num_graphs);
        for (size_t run = 0; run < num_graphs; ++run) {
            all_measurements[run].resize(N);
            for (size_t k = 0; k < N; ++k) {
                size_t idx = run * N * 2 + k * 2;
                Eigen::Vector2d meas;
                meas << buffer[idx + 0], buffer[idx + 1];
                all_measurements[run][k] = meas;
            }
        }
    } catch (H5::Exception& e) {
        std::cerr << "Error loading measurements from HDF5: " << e.getDetailMsg() << std::endl;
    }
    return all_measurements;
} 
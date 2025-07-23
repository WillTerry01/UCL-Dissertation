#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>
#include <H5Cpp.h>

using namespace std;
using namespace g2o;

FactorGraph2DTrajectory::FactorGraph2DTrajectory() : N_(0), chi2_(0.0) {
    Q_ = Eigen::Matrix4d::Identity() * 0.1;
    R_ = Eigen::Matrix2d::Identity() * 0.5;
}

void FactorGraph2DTrajectory::run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, bool add_process_noise) {
    N_ = static_cast<int>(true_states.size());
    true_states_ = true_states;
    if (add_process_noise && N_ > 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise_q(0.0, sqrt(Q_(0,0)));
        for (int k = 1; k < N_; ++k) {
            Eigen::Vector4d process_noise;
            process_noise << noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen);
            true_states_[k] += process_noise;
        }
    }
    measurements_.resize(N_);
    if (measurements) {
        measurements_ = *measurements;
    } else {
        // Generate noisy measurements from true states using R_
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise_p(0.0, sqrt(R_(0,0)));
        for (int k = 0; k < N_; ++k) {
            measurements_[k] = true_states_[k].head<2>() + Eigen::Vector2d(noise_p(gen), noise_p(gen));
        }
    }

    // //print the q and r parameters
    // std::cout << "Q: " << Q_ << std::endl;
    // std::cout << "R: " << R_ << std::endl;ers
    // std::cout << "Q: " << Q_ << std::endl;
    // std::cout << "R: " << R_ << std::e

    setupOptimizer();
    optimize();
}

void FactorGraph2DTrajectory::setupOptimizer() {
    typedef BlockSolver< BlockSolverTraits<4, 4> > BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    blockSolver_ = blockSolver.get();
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer_ = std::make_unique<SparseOptimizer>();
    optimizer_->setAlgorithm(solver);
    optimizer_->setVerbose(false);

    vertices_.resize(N_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_q(0.0, sqrt(Q_(0,0)));
    for (int k = 0; k < N_; ++k) {
        vertices_[k] = new Vertex4D();
        vertices_[k]->setId(k);
        vertices_[k]->setEstimate(true_states_[k] + Eigen::Vector4d(noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen)));
        optimizer_->addVertex(vertices_[k]);
    }
    
    // Use dt = 1.0 for now (can be made configurable later)
    double dt = 1.0;
    for (int k = 1; k < N_; ++k) {
        EdgeProcessModel* e = new EdgeProcessModel(dt);
        e->setVertex(0, vertices_[k-1]);
        e->setVertex(1, vertices_[k]);
        e->setMeasurement(Eigen::Vector4d::Zero());
        e->setInformation(Q_.inverse());
        optimizer_->addEdge(e);
    }
    for (int k = 0; k < N_; ++k) {
        EdgeMeasurement* e = new EdgeMeasurement();
        e->setVertex(0, vertices_[k]);
        e->setMeasurement(measurements_[k]);
        e->setInformation(R_.inverse());
        optimizer_->addEdge(e);
    }
}

void FactorGraph2DTrajectory::optimize() {
    optimizer_->initializeOptimization();
    optimizer_->optimize(20);
    chi2_ = optimizer_->chi2();
}

double FactorGraph2DTrajectory::getChi2() const {
    return chi2_;
}

void FactorGraph2DTrajectory::writeCSV(const std::string& filename) const {
    std::ofstream csv(filename);
    csv << "t,true_x,true_y,meas_x,meas_y,est_x,est_y\n";
    for (int k = 0; k < N_; ++k) {
        const auto& true_state = true_states_[k];
        const auto& meas = measurements_[k];
        const auto& v = vertices_[k]->estimate();
        csv << k << ","
            << true_state[0] << "," << true_state[1] << ","
            << meas[0] << "," << meas[1] << ","
            << v[0] << "," << v[1] << "\n";
    }
    csv << "chi2," << chi2_ << "\n";
    csv.close();
}

void FactorGraph2DTrajectory::writeHDF5(const std::string& filename) const {
    using namespace H5;
    const hsize_t N = N_;
    const hsize_t D = 7; // t, true_x, true_y, meas_x, meas_y, est_x, est_y
    std::vector<double> data(N * D);
    for (int k = 0; k < N_; ++k) {
        const auto& true_state = true_states_[k];
        const auto& meas = measurements_[k];
        const auto& v = vertices_[k]->estimate();
        data[k * D + 0] = static_cast<double>(k);
        data[k * D + 1] = true_state[0];
        data[k * D + 2] = true_state[1];
        data[k * D + 3] = meas[0];
        data[k * D + 4] = meas[1];
        data[k * D + 5] = v[0];
        data[k * D + 6] = v[1];
    }
    H5File file(filename, H5F_ACC_TRUNC);
    hsize_t dims[2] = {N, D};
    DataSpace dataspace(2, dims);
    DataSet dataset = file.createDataSet("trajectory", PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(data.data(), PredType::NATIVE_DOUBLE);
    // Write chi2 as an attribute
    Attribute chi2_attr = dataset.createAttribute("chi2", PredType::NATIVE_DOUBLE, DataSpace());
    chi2_attr.write(PredType::NATIVE_DOUBLE, &chi2_);
}

void FactorGraph2DTrajectory::printHessian() const {
    if (!blockSolver_) {
        std::cout << "BlockSolver not initialized." << std::endl;
        return;
    }
    const auto* hessian = blockSolver_->hessian();
    if (!hessian) {
        std::cout << "Hessian is null." << std::endl;
        return;
    }
    std::cout << "Hessian (pose-pose block):" << std::endl;
    for (size_t i = 0; i < hessian->blockCols().size(); ++i) {
        for (const auto& blockPair : hessian->blockCols()[i]) {
            int row = blockPair.first;
            const auto* block = blockPair.second;
            std::cout << "Block (" << row << ", " << i << "):\n" << *block << "\n";
        }
    }
}

Eigen::MatrixXd FactorGraph2DTrajectory::getFullInformationMatrix() const {
    if (!blockSolver_ || !blockSolver_->hessian()) {
        std::cerr << "Hessian not available!" << std::endl;
        return Eigen::MatrixXd();
    }
    const auto* hessian = blockSolver_->hessian();
    
    // Get the total dimension (number of vertices * vertex dimension)
    int total_dim = N_ * 4;  // Each vertex has dimension 4
    Eigen::MatrixXd fullHessian = Eigen::MatrixXd::Zero(total_dim, total_dim);
    
    for (size_t i = 0; i < hessian->blockCols().size(); ++i) {
        for (const auto& blockPair : hessian->blockCols()[i]) {
            int row = blockPair.first;
            const auto* block = blockPair.second;
            if (block) {
                fullHessian.block(row * 4, i * 4, 4, 4) = *block;
            }
        }
    }
    return fullHessian;
}

void FactorGraph2DTrajectory::setQFromProcessNoiseIntensity(double q_intensity, double dt) {
    // Construct the proper Q matrix for 2D linear tracking
    // Q = [dt^3/3 * V₀    0           dt^2/2 * V₀    0        ]
    //     [0               dt^3/3 * V₁ 0               dt^2/2 * V₁]
    //     [dt^2/2 * V₀    0           dt * V₀         0        ]
    //     [0               dt^2/2 * V₁ 0               dt * V₁  ]
    Q_ = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    
    // Use same noise intensity for both x and y directions (V₀ = V₁ = q_intensity)
    double V0 = q_intensity;
    double V1 = q_intensity;
    
    // Add small numerical tolerance to ensure positive definiteness
    double numerical_tolerance = 1e-10;
    
    // Position-position covariance (diagonal)
    Q_(0, 0) = dt3 / 3.0 * V0 + numerical_tolerance;  // x position variance
    Q_(1, 1) = dt3 / 3.0 * V1 + numerical_tolerance;  // y position variance
    
    // Velocity-velocity covariance (diagonal)
    Q_(2, 2) = dt * V0 + numerical_tolerance;         // x velocity variance
    Q_(3, 3) = dt * V1 + numerical_tolerance;         // y velocity variance
    
    // Position-velocity cross covariance
    Q_(0, 2) = dt2 / 2.0 * V0;  // x position - x velocity covariance
    Q_(2, 0) = Q_(0, 2);        // symmetric
    Q_(1, 3) = dt2 / 2.0 * V1;  // y position - y velocity covariance
    Q_(3, 1) = Q_(1, 3);        // symmetric
}

void FactorGraph2DTrajectory::setRFromMeasurementNoise(double sigma_x, double sigma_y) {
    // Set R matrix for measurement noise with different variances for x and y
    R_ = Eigen::Matrix2d::Zero();
    R_(0, 0) = sigma_x * sigma_x;  // x measurement variance
    R_(1, 1) = sigma_y * sigma_y;  // y measurement variance
}

void FactorGraph2DTrajectory::setQFromScalar(double q_scalar, double dt) {
    // Legacy method: set Q as scalar multiple of identity
    Q_ = Eigen::Matrix4d::Identity() * q_scalar;
}

void FactorGraph2DTrajectory::setRFromScalar(double r_scalar) {
    // Legacy method: set R as scalar multiple of identity
    R_ = Eigen::Matrix2d::Identity() * r_scalar;
} 
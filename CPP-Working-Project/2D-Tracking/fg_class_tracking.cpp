#include "fg_class_tracking.h"
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>
#include <H5Cpp.h>

using namespace std;
using namespace g2o;

FactorGraph2DTrajectory::FactorGraph2DTrajectory() : N_(0), chi2_(0.0) {
    // Q_ and R_ will be set by setQFromProcessNoiseIntensity() and setRFromMeasurementNoise()
    // No need to initialize with legacy values
}

void FactorGraph2DTrajectory::run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, bool add_process_noise, double dt) {
    // Check if Q and R matrices are properly initialized
    if (Q_.isZero()) {
        throw std::runtime_error("Q matrix is not initialized. Please call setQFromProcessNoiseIntensity() or set Q_ directly before calling run().");
    }
    if (R_.isZero()) {
        throw std::runtime_error("R matrix is not initialized. Please call setRFromMeasurementNoise() or set R_ directly before calling run().");
    }
    
    // Check if matrices are positive definite (required for inverse)
    Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q_);
    if (lltOfQ.info() != Eigen::Success) {
        throw std::runtime_error("Q matrix is not positive definite. Cannot compute inverse for factor graph.");
    }
    
    Eigen::LLT<Eigen::Matrix2d> lltOfR(R_);
    if (lltOfR.info() != Eigen::Success) {
        throw std::runtime_error("R matrix is not positive definite. Cannot compute inverse for factor graph.");
    }
    
    N_ = true_states.size();
    true_states_ = true_states;
    dt_ = dt;  // Store the dt parameter
    
    if (measurements) {
        measurements_ = *measurements;
    } else {
        // Generate measurements from true states if not provided
        measurements_.resize(N_);
        // Use a more deterministic seed based on memory address and time
        std::mt19937 gen(54321 + reinterpret_cast<uintptr_t>(this));  // Deterministic seed
        std::normal_distribution<> noise_r(0.0, 0.1);
        for (int k = 0; k < N_; ++k) {
            measurements_[k][0] = true_states_[k][0] + noise_r(gen);
            measurements_[k][1] = true_states_[k][1] + noise_r(gen);
        }
    }
    
    if (add_process_noise) {
        // Add process noise to true states
        // Use a more deterministic seed based on memory address
        std::mt19937 gen(98765 + reinterpret_cast<uintptr_t>(this));  // Deterministic seed
        std::normal_distribution<> noise_q(0.0, 0.1);
        for (int k = 1; k < N_; ++k) {
            true_states_[k] += Eigen::Vector4d(noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen));
        }
    }
    
    setupOptimizer();
    
    // Initialize vertices with noisy true states
    // Use deterministic seeding to avoid thread-dependent initialization issues
    // Use a more deterministic seed based on memory address
    std::mt19937 gen(12345 + reinterpret_cast<uintptr_t>(this));  // Deterministic seed
    
    // Use proper multivariate initialization noise based on full Q matrix
    // Check if Q matrix is positive definite for Cholesky decomposition
    if (lltOfQ.info() == Eigen::Success) {
        // Use Cholesky decomposition to sample from full Q covariance structure
        Eigen::Matrix4d L = lltOfQ.matrixL();
        std::normal_distribution<> standard_normal(0.0, 1.0);
        
        for (int k = 0; k < N_; ++k) {
            // Generate uncorrelated standard normal noise
            Eigen::Vector4d uncorrelated_noise;
            for (int i = 0; i < 4; ++i) {
                uncorrelated_noise[i] = standard_normal(gen);
            }
            
            // Transform to correlated noise using Q = L*L^T
            Eigen::Vector4d correlated_noise = L * uncorrelated_noise;
            
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(true_states_[k] + correlated_noise);
            optimizer_->addVertex(vertices_[k]);
        }
    } else {
        std::cout << "Q matrix is not positive definite. Using fallback initialization." << std::endl;
        // Fallback to diagonal initialization if Q is not positive definite
        double init_noise_std = std::max(0.01, std::sqrt(std::max(Q_(0,0), Q_(1,1)) + std::max(Q_(2,2), Q_(3,3))));
        std::normal_distribution<> noise_q(0.0, init_noise_std);
        for (int k = 0; k < N_; ++k) {
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(true_states_[k] + Eigen::Vector4d(noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen)));
            optimizer_->addVertex(vertices_[k]);
        }
    }
    
    // Use the dt parameter passed to the method
    for (int k = 1; k < N_; ++k) {
        EdgeProcessModel* e = new EdgeProcessModel(dt_);
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

    // Create vertices but don't add them yet - they will be added in run()
    vertices_.resize(N_);
    for (int k = 0; k < N_; ++k) {
        vertices_[k] = new Vertex4D();
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

Eigen::MatrixXd FactorGraph2DTrajectory::getFullHessianMatrix() const {
    if (!blockSolver_ || !blockSolver_->hessian()) {
        std::cerr << "Hessian not available!" << std::endl;
        return Eigen::MatrixXd();
    }
    
    // Get the total dimension (number of vertices * vertex dimension)
    int total_dim = N_ * 4;  // Each vertex has dimension 4
    
    // Try to get the full Hessian directly from g2o
    // First, let's try the block-wise approach but more carefully
    const auto* hessian = blockSolver_->hessian();
    Eigen::MatrixXd fullHessian = Eigen::MatrixXd::Zero(total_dim, total_dim);
    
    // Fill in the existing blocks
    for (size_t i = 0; i < hessian->blockCols().size(); ++i) {
        for (const auto& blockPair : hessian->blockCols()[i]) {
            int row = blockPair.first;
            const auto* block = blockPair.second;
            if (block && row < N_ && i < N_) {  // Safety check
                fullHessian.block(row * 4, i * 4, 4, 4) = *block;
            }
        }
    }
    
    // Ensure the matrix is symmetric by copying upper triangle to lower triangle
    for (int i = 0; i < total_dim; ++i) {
        for (int j = i + 1; j < total_dim; ++j) {
            fullHessian(j, i) = fullHessian(i, j);
        }
    }
    
    // Add small diagonal regularization to ensure positive definiteness
    double regularization = 1e-12;
    for (int i = 0; i < total_dim; ++i) {
        fullHessian(i, i) += regularization;
    }
    
    return fullHessian;
}

std::vector<Eigen::Vector4d> FactorGraph2DTrajectory::getEstimatesInternal() const {
    std::vector<Eigen::Vector4d> estimates(N_);
    for (int k = 0; k < N_; ++k) {
        estimates[k] = vertices_[k]->estimate();
    }
    return estimates;
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
    
    // Add numerical regularization to ensure positive definiteness and invertibility
    double min_variance = 1e-8;  // Minimum variance to prevent singularity
    V0 = std::max(V0, min_variance / dt);  // Ensure minimum process noise intensity
    V1 = std::max(V1, min_variance / dt);
    
    // Position-position covariance (diagonal)
    Q_(0, 0) = dt3 / 3.0 * V0;  // x position variance
    Q_(1, 1) = dt3 / 3.0 * V1;  // y position variance
    
    // Velocity-velocity covariance (diagonal)
    Q_(2, 2) = dt * V0;         // x velocity variance
    Q_(3, 3) = dt * V1;         // y velocity variance
    
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

 
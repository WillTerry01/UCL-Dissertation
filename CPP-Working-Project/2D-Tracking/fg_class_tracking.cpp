/*
 * 2D Factor Graph Trajectory Estimation
 * =====================================
 * 
 * PURPOSE:
 * This class implements a factor graph-based estimator for 2D tracking problems.
 * It tests propositions 3 and 4 from factor graph theory using g2o optimization
 * framework to validate normalized innovation squared (NIS) distributions.
 * 
 * INPUT DATA:
 * - true_states: Noisy trajectory data from tracking_gen_data.cpp [x, y, vx, vy]
 * - measurements: Noisy position observations [x_obs, y_obs] 
 * - Q matrix: Process noise covariance (set via setQFromProcessNoiseIntensity())
 * - R matrix: Measurement noise covariance (set via setRFromMeasurementNoise())
 * 
 * FACTOR GRAPH STRUCTURE:
 * - Vertices: State estimates at each timestep [x, y, vx, vy]
 * - Process Edges: Connect consecutive states using constant velocity model
 * - Measurement Edges: Connect states to position observations
 * - Information matrices: Q^{-1} for process edges, R^{-1} for measurement edges
 * 
 * TESTING MODES (following MATLAB research implementation):
 * 
 * NIS3 (Proposition 3): do_optimization = false
 * - Initialize: Exactly at noisy ground truth from data generation
 * - Optimization: DISABLED - no optimization performed
 * - Purpose: Test graph structure validity with perfect initialization
 * - Chi-squared: Measures residual errors with ideal state estimates
 * 
 * NIS4 (Proposition 4): do_optimization = true  
 * - Initialize: At ZERO (challenging starting point, matches MATLAB approach)
 * - Optimization: ENABLED - Levenberg-Marquardt optimization
 * - Purpose: Test optimization robustness and convergence from poor initialization
 * - Chi-squared: Measures final estimation errors after optimization
 * 
 * OUTPUTS:
 * - Chi-squared values for statistical analysis
 * - Final state estimates after optimization (if enabled)
 * - CSV/HDF5 files with trajectory comparison data
 * - Full Hessian matrix for uncertainty analysis
 * 
 * KEY DESIGN DECISIONS:
 * - NO process noise addition: Input states already contain realistic noise
 * - Zero initialization for NIS4: Matches validated MATLAB implementation  
 * - Positive definite enforcement: Throws errors for invalid Q/R matrices
 * - Mandatory measurements: No synthetic measurement generation
 * 
 * This implementation eliminates double noise application and follows the exact
 * approach from published factor graph research for robust statistical validation.
 */

#include "fg_class_tracking.h"
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>
#include <H5Cpp.h>

using namespace std;
using namespace g2o;

FactorGraph2DTrajectory::FactorGraph2DTrajectory() : N_(0), chi2_(0.0), motion_model_type_("linear"), measurement_model_type_("linear"), turn_rate_(0.0) {
    // Q_ and R_ will be set by setQFromProcessNoiseIntensity() and setRFromMeasurementNoise()
    // No need to initialize with legacy values
}

void FactorGraph2DTrajectory::run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, double dt, bool do_optimization) {
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
    
    // Check if measurements are provided
    if (measurements) {
        measurements_ = *measurements;
    } else {
        throw std::runtime_error("Measurements are required. Please provide a valid measurements vector.");
    }
    
    setupOptimizer();
    
    // Initialization strategy based on MATLAB student approach
    if (!do_optimization) {
        // Proposition 3 (NIS3): Initialize exactly at ground truth (no noise)
        for (int k = 0; k < N_; ++k) {
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(true_states_[k]);  // Exact ground truth
            optimizer_->addVertex(vertices_[k]);
        }
    } else {
        // Proposition 4 (NIS4): Initialize at zero (matching MATLAB approach)
        // MATLAB uses: v{n}.setEstimate(0*trueX(:, n))
        for (int k = 0; k < N_; ++k) {
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(Eigen::Vector4d::Zero());  // Initialize at zero
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
    // Add measurement edges - both linear and nonlinear use GPS measurements
    for (int k = 0; k < N_; ++k) {
        EdgeGPSMeasurement* e = new EdgeGPSMeasurement();
        e->setVertex(0, vertices_[k]);
        e->setMeasurement(measurements_[k]);
        e->setInformation(R_.inverse());
        optimizer_->addEdge(e);
    }
    
        // Only optimize if requested (Proposition 4), skip for Proposition 3
    if (do_optimization) {
        optimize();
    } else {
        // For Proposition 3, calculate chi2 WITHOUT optimization (per MATLAB reference)
        // Since g2o's chi2() requires optimization, we manually calculate chi2 from residuals
        optimizer_->initializeOptimization();
        
        // Manually compute chi2 by iterating through all edges and summing their contributions
        chi2_ = 0.0;
        for (auto it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it) {
            g2o::OptimizableGraph::Edge* edge = static_cast<g2o::OptimizableGraph::Edge*>(*it);
            edge->computeError();  // Compute error vector for this edge
            
            // Use g2o's built-in chi2() method for individual edges
            // This avoids direct access to error vectors and information matrices
            double edge_chi2 = edge->chi2();
            chi2_ += edge_chi2;
        }
    }
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

// New methods to calculate actual graph dimensions (like MATLAB student implementation)
int FactorGraph2DTrajectory::getActualGraphDimZ() const {
    if (!optimizer_) {
        std::cerr << "Error: Optimizer not initialized" << std::endl;
        return 0;
    }
    
    int dimZ = 0;
    // Sum dimensions of all edges (both process and measurement edges)
    for (auto it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it) {
        g2o::OptimizableGraph::Edge* edge = static_cast<g2o::OptimizableGraph::Edge*>(*it);
        dimZ += edge->dimension();
    }
    return dimZ;
}

int FactorGraph2DTrajectory::getActualGraphDimX() const {
    if (!optimizer_) {
        std::cerr << "Error: Optimizer not initialized" << std::endl;
        return 0;
    }
    
    int dimX = 0;
    // Sum dimensions of all vertices
    for (auto it = optimizer_->vertices().begin(); it != optimizer_->vertices().end(); ++it) {
        g2o::OptimizableGraph::Vertex* vertex = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
        dimX += vertex->dimension();
    }
    return dimX;
}

std::pair<int, int> FactorGraph2DTrajectory::getActualGraphDimensions() const {
    return std::make_pair(getActualGraphDimZ(), getActualGraphDimX());
}

// Nonlinear system method implementations
void FactorGraph2DTrajectory::setMotionModelType(const std::string& model_type, double turn_rate) {
    motion_model_type_ = model_type;
    turn_rate_ = turn_rate;
}

void FactorGraph2DTrajectory::setMeasurementModelType(const std::string& model_type) {
    measurement_model_type_ = model_type;
}

void FactorGraph2DTrajectory::runNonlinear(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, double dt, bool do_optimization) {
    // Check if Q and R matrices are properly initialized
    if (Q_.isZero()) {
        throw std::runtime_error("Q matrix is not initialized. Please call setQFromProcessNoiseIntensity() or set Q_ directly before calling runNonlinear().");
    }
    if (R_.isZero()) {
        throw std::runtime_error("R matrix is not initialized. Please call setRFromMeasurementNoise() or set R_ directly before calling runNonlinear().");
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
    dt_ = dt;
    
    // Check if measurements are provided
    if (measurements) {
        measurements_ = *measurements;
    } else {
        throw std::runtime_error("Measurements are required. Please provide a valid measurements vector.");
    }
    
    setupOptimizer();
    
    // Initialization strategy based on MATLAB student approach
    if (!do_optimization) {
        // Proposition 3 (NIS3): Initialize exactly at ground truth (no noise)
        for (int k = 0; k < N_; ++k) {
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(true_states_[k]);  // Exact ground truth
            optimizer_->addVertex(vertices_[k]);
        }
    } else {
        // Proposition 4 (NIS4): Initialize at zero (matching MATLAB approach)
        for (int k = 0; k < N_; ++k) {
            vertices_[k]->setId(k);
            vertices_[k]->setEstimate(Eigen::Vector4d::Zero());  // Initialize at zero
            optimizer_->addVertex(vertices_[k]);
        }
    }
    
    // Add motion model edges based on selected model type
    for (int k = 1; k < N_; ++k) {
        if (motion_model_type_ == "linear") {
            EdgeProcessModel* e = new EdgeProcessModel(dt_);
            e->setVertex(0, vertices_[k-1]);
            e->setVertex(1, vertices_[k]);
            e->setMeasurement(Eigen::Vector4d::Zero());
            e->setInformation(Q_.inverse());
            optimizer_->addEdge(e);
        } else if (motion_model_type_ == "constant_turn_rate") {
            EdgeNonlinearMotionCT* e = new EdgeNonlinearMotionCT(dt_, turn_rate_);
            e->setVertex(0, vertices_[k-1]);
            e->setVertex(1, vertices_[k]);
            e->setMeasurement(Eigen::Vector4d::Zero());
            e->setInformation(Q_.inverse());
            optimizer_->addEdge(e);
        } else {
            throw std::runtime_error("Unsupported motion model type: " + motion_model_type_);
        }
    }
    
    // Add measurement edges based on selected model type
    for (int k = 0; k < N_; ++k) {
        if (measurement_model_type_ == "linear" || measurement_model_type_ == "gps") {
            // Both linear and GPS models use Cartesian position measurements (x, y)
            EdgeGPSMeasurement* e = new EdgeGPSMeasurement();
            e->setVertex(0, vertices_[k]);
            e->setMeasurement(measurements_[k]);
            e->setInformation(R_.inverse());
            optimizer_->addEdge(e);
        } else {
            throw std::runtime_error("Unsupported measurement model type: " + measurement_model_type_);
        }
    }
    
    // Only optimize if requested (Proposition 4), skip for Proposition 3
    if (do_optimization) {
        optimize();
    } else {
        // For Proposition 3, calculate chi2 WITHOUT optimization (per MATLAB reference)
        optimizer_->initializeOptimization();
        
        // Manually compute chi2 by iterating through all edges and summing their contributions
        chi2_ = 0.0;
        for (auto it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it) {
            g2o::OptimizableGraph::Edge* edge = static_cast<g2o::OptimizableGraph::Edge*>(*it);
            edge->computeError();  // Compute error vector for this edge
            
            // Use g2o's built-in chi2() method for individual edges
            double edge_chi2 = edge->chi2();
            chi2_ += edge_chi2;
        }
    }
}

 
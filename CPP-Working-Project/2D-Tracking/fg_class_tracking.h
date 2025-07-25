#pragma once
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/optimizable_graph.h>
#include <H5Cpp.h>

// Vertex: 2D position and velocity [x, y, vx, vy]
class Vertex4D : public g2o::BaseVertex<4, Eigen::Vector4d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate = Eigen::Vector4d::Zero(); }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector4d(update[0], update[1], update[2], update[3]);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Process model edge: connects two consecutive states
class EdgeProcessModel : public g2o::BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructor to set dt and control input
    EdgeProcessModel(double dt = 1.0) : dt_(dt) {}
    
    void computeError() override {
        const Vertex4D* v1 = static_cast<const Vertex4D*>(_vertices[0]);
        const Vertex4D* v2 = static_cast<const Vertex4D*>(_vertices[1]);
        
        // State transition matrix F (constant velocity model)
        // F = [1  0  dt  0]
        //     [0  1  0   dt]
        //     [0  0  1   0]
        //     [0  0  0   1]
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0, 2) = dt_;  // x position += x velocity * dt
        F(1, 3) = dt_;  // y position += y velocity * dt
        
        // Control input matrix B (for future acceleration control)
        // B = [0.5 * dt^2  0        ]
        //     [0           0.5 * dt^2]
        //     [dt          0        ]
        //     [0           dt       ]
        Eigen::Matrix<double, 4, 2> B;
        double dt2 = dt_ * dt_;
        B << 0.5 * dt2, 0.0,
             0.0, 0.5 * dt2,
             dt_, 0.0,
             0.0, dt_;
        
        // Get control input (acceleration) - set to zero for constant velocity
        // TODO: Modify this to get actual control input for controlled motion
        Eigen::Vector2d acceleration = getControlInput();  // Currently returns zero
        
        // State equation: xₖ₊₁ = Fxₖ + Buₖ + vₖ
        Eigen::Vector4d control_effect = B * acceleration;
        Eigen::Vector4d pred = F * v1->estimate() + control_effect;
        
        // Error: e = xₖ₊₁ - (Fxₖ + Buₖ)
        _error = v2->estimate() - pred;
    }
    
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    
private:
    double dt_;
    
    // Control input generation function - currently returns zero for constant velocity
    // TODO: Modify this function to return actual control input for controlled motion
    Eigen::Vector2d getControlInput() {
        // For now, return zero acceleration (constant velocity)
        // This can be modified later to return:
        // - Sinusoidal acceleration: [A*sin(ω*t), A*cos(ω*t)]
        // - Step acceleration: [a_x, a_y] for t > t_switch
        // - Random acceleration: [N(0,σ²), N(0,σ²)]
        // - Nonlinear control laws
        return Eigen::Vector2d::Zero();
    }
};

// Measurement edge: connects state to observed position
class EdgeMeasurement : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex4D* v = static_cast<const Vertex4D*>(_vertices[0]);
        _error = _measurement - v->estimate().head<2>();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class FactorGraph2DTrajectory {
public:
    struct OutputOptions {
        bool output_estimated_state = false;
        bool output_true_state = false;
        bool output_information_matrix = false;
    };

    FactorGraph2DTrajectory();
    void setOutputOptions(const OutputOptions& options) { output_options_ = options; }
    OutputOptions getOutputOptions() const { return output_options_; }
    // Get all estimated states
    std::vector<Eigen::Vector4d> getAllEstimates() const { return getEstimatesInternal(); }
    // Get all true states
    std::vector<Eigen::Vector4d> getAllTrueStates() const { return true_states_; }
    // Get the full information matrix (Hessian)
    Eigen::MatrixXd getFullHessianMatrix() const;
    // Run the factor graph optimization with user-provided true states and optional measurements.
    // If add_process_noise is true, process noise will be added to the true states inside the function.
    void run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements = nullptr, bool add_process_noise = false, double dt = 1.0);
    void writeCSV(const std::string& filename = "../H5_Files/2d_trajectory_estimate.csv") const;
    void writeHDF5(const std::string& filename = "../H5_Files/2d_trajectory_estimate.h5") const;
    double getChi2() const;
    // Getter for estimated state at time step k
    Eigen::Vector4d getEstimate(int k) const { return vertices_[k]->estimate(); }
    void printHessian() const;

    // Add these new methods
    void setQFromProcessNoiseIntensity(double q_intensity, double dt = 1.0);
    void setRFromMeasurementNoise(double sigma_x, double sigma_y);

private:
    std::vector<Vertex4D*> vertices_;
    std::vector<Eigen::Vector4d> true_states_;
    std::vector<Eigen::Vector2d> measurements_;
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
    g2o::BlockSolver<g2o::BlockSolverTraits<4, 4>>* blockSolver_;
    int N_;
    double chi2_;
    Eigen::Matrix4d Q_;
    Eigen::Matrix2d R_;
    OutputOptions output_options_;
    double dt_;  // Add dt as a member variable
    
    // Helper method to extract optimized estimates from vertices
    std::vector<Eigen::Vector4d> getEstimatesInternal() const;
    void setupOptimizer();
    void optimize();
}; 
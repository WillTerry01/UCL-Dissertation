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
#include <g2o/core/marginal_covariance_cholesky.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/optimizable_graph.h>

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
    void computeError() override {
        const Vertex4D* v1 = static_cast<const Vertex4D*>(_vertices[0]);
        const Vertex4D* v2 = static_cast<const Vertex4D*>(_vertices[1]);
        double dt = 1.0;
        Eigen::Vector4d pred;
        pred.head<2>() = v1->estimate().head<2>() + v1->estimate().tail<2>() * dt;
        pred.tail<2>() = v1->estimate().tail<2>();
        _error = v2->estimate() - pred;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
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
    FactorGraph2DTrajectory();
    // Q_ and R_ are public for direct modification
    Eigen::Matrix4d Q_;
    Eigen::Matrix2d R_;
    // Run the factor graph optimization with user-provided true states and optional measurements.
    // If add_process_noise is true, process noise will be added to the true states inside the function.
    void run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements = nullptr, bool add_process_noise = false);
    void writeCSV(const std::string& filename = "2d_trajectory_estimate.csv") const;
    double getChi2() const;
    // Optionally, add getters for results
private:
    int N_;
    std::vector<Eigen::Vector4d> true_states_;
    std::vector<Eigen::Vector2d> measurements_;
    std::vector<Vertex4D*> vertices_;
    double chi2_;
    void setupOptimizer();
    void optimize();
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
}; 
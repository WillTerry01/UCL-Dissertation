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

// Vertex: 1D position and velocity [x, x_dot]
class Vertex2D : public g2o::BaseVertex<2, Eigen::Vector2d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override { _estimate = Eigen::Vector2d::Zero(); }
    void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector2d(update[0], update[1]);
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Process model edge: connects two consecutive states
class EdgeProcessModel1D : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, Vertex2D, Vertex2D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex2D* v1 = static_cast<const Vertex2D*>(_vertices[0]);
        const Vertex2D* v2 = static_cast<const Vertex2D*>(_vertices[1]);
        double dt = 1.0;
        Eigen::Vector2d pred;
        pred[0] = v1->estimate()[0] + v1->estimate()[1] * dt;
        pred[1] = v1->estimate()[1];
        _error = v2->estimate() - pred;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Measurement edge: connects state to observed position
class EdgeMeasurement1D : public g2o::BaseUnaryEdge<1, double, Vertex2D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex2D* v = static_cast<const Vertex2D*>(_vertices[0]);
        _error[0] = _measurement - v->estimate()[0];
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class FactorGraph1DTrajectory {
public:
    FactorGraph1DTrajectory();
    Eigen::Matrix2d Q_;
    Eigen::Matrix<double, 1, 1> R_;
    void run(const std::vector<Eigen::Vector2d>& true_states, const std::vector<double>* measurements = nullptr, bool add_process_noise = false);
    void writeCSV(const std::string& filename = "1d_trajectory_estimate.csv") const;
    double getChi2() const;
private:
    int N_;
    std::vector<Eigen::Vector2d> true_states_;
    std::vector<double> measurements_;
    std::vector<Vertex2D*> vertices_;
    double chi2_;
    void setupOptimizer();
    void optimize();
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
}; 
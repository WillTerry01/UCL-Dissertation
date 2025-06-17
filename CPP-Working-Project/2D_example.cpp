#include <iostream>
#include <fstream>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <memory>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace g2o;

// Define a 4D vertex (state: position and velocity)
class Vertex4D : public BaseVertex<4, Eigen::Vector4d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector4d::Zero();
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector4d(update[0], update[1], update[2], update[3]);
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

// Unary edge representing a measurement of the 2D position
class UnaryEdge2D : public BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError() override {
        const Vertex4D* v = static_cast<const Vertex4D*>(_vertices[0]);
        // Only use position part of state for measurement
        _error = _measurement - v->estimate().head<2>();
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

int main() {
    // Setup the optimizer
    typedef BlockSolver< BlockSolverTraits<4, 2> > BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Process noise covariance (Q) and measurement noise covariance (R)
    Eigen::Matrix4d Q = Eigen::Matrix4d::Identity() * 0.1;  // Process noise
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 1.0;  // Measurement noise

    // Add vertex (state)
    Vertex4D* v = new Vertex4D();
    v->setId(0);
    v->setEstimate(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0));  // initial state: [x, y, vx, vy]
    optimizer.addVertex(v);

    // Generate noisy measurements around a true position
    const Eigen::Vector2d true_position(5.0, 3.0);
    const Eigen::Vector2d true_velocity(0.5, 0.3);  // Constant velocity
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 1.0);

    // Add measurements
    std::ofstream csv("build/2d_progress.csv");
    csv << "iteration,x,y,vx,vy,total_error\n";

    // Generate 10 noisy measurements
    for (int i = 0; i < 10; ++i) {
        // Update true position based on velocity
        Eigen::Vector2d measurement = true_position + true_velocity * i;
        measurement.x() += noise(gen);
        measurement.y() += noise(gen);

        UnaryEdge2D* e = new UnaryEdge2D();
        e->setVertex(0, v);
        e->setMeasurement(measurement);
        e->setInformation(R.inverse());  // Use measurement noise matrix
        e->setId(i);
        optimizer.addEdge(e);

        // Add process noise edge
        if (i > 0) {
            // Create a binary edge for process model
            // This would require implementing a new edge type for the process model
            // For simplicity, we'll just add process noise to the state directly
            Eigen::Vector4d process_noise;
            process_noise << noise(gen), noise(gen), noise(gen), noise(gen);
            process_noise = Q * process_noise;
            v->setEstimate(v->estimate() + process_noise);
        }
    }

    // Optimize step by step and log progress
    optimizer.initializeOptimization();
    int maxIterations = 20;
    for (int iter = 0; iter <= maxIterations; ++iter) {
        if (iter > 0) optimizer.optimize(1);
        
        // Compute total error
        double totalError = 0.0;
        for (const auto& edge : optimizer.edges()) {
            UnaryEdge2D* e = static_cast<UnaryEdge2D*>(edge);
            e->computeError();
            totalError += e->error().squaredNorm();
        }

        // Log progress
        csv << iter << "," 
            << v->estimate().x() << "," 
            << v->estimate().y() << "," 
            << v->estimate().z() << "," 
            << v->estimate().w() << "," 
            << totalError << "\n";

        std::cout << "Iter " << iter 
                  << ": Position = (" << v->estimate().x() << ", " << v->estimate().y() 
                  << "), Velocity = (" << v->estimate().z() << ", " << v->estimate().w()
                  << "), Error = " << totalError << std::endl;
    }
    csv.close();

    // Output final result
    std::cout << "Final optimized state: [" 
              << v->estimate().x() << ", " << v->estimate().y() << ", "
              << v->estimate().z() << ", " << v->estimate().w() << "]" << std::endl;
    std::cout << "True position: (" 
              << true_position.x() << ", " << true_position.y() << ")" << std::endl;
    std::cout << "True velocity: (" 
              << true_velocity.x() << ", " << true_velocity.y() << ")" << std::endl;

    return 0;
}

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

using namespace std;
using namespace g2o;

// Define a 2D vertex (state)
class Vertex2D : public BaseVertex<2, Eigen::Vector2d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector2d::Zero();
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector2d(update[0], update[1]);
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

// Unary edge representing a measurement of the 2D position
class UnaryEdge2D : public BaseUnaryEdge<2, Eigen::Vector2d, Vertex2D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError() override {
        const Vertex2D* v = static_cast<const Vertex2D*>(_vertices[0]);
        _error = _measurement - v->estimate();
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

int main() {
    // Setup the optimizer
    typedef BlockSolver< BlockSolverTraits<2, 2> > BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Add vertex (state)
    Vertex2D* v = new Vertex2D();
    v->setId(0);
    v->setEstimate(Eigen::Vector2d(0.0, 0.0));  // initial guess
    optimizer.addVertex(v);

    // Generate noisy measurements around a true position
    const Eigen::Vector2d true_position(5.0, 3.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.5);  // Gaussian noise with std dev 0.5

    // Add measurements
    std::ofstream csv("build/2d_progress.csv");
    csv << "iteration,x,y,total_error\n";

    // Generate 10 noisy measurements
    for (int i = 0; i < 10; ++i) {
        Eigen::Vector2d measurement = true_position;
        measurement.x() += noise(gen);
        measurement.y() += noise(gen);

        UnaryEdge2D* e = new UnaryEdge2D();
        e->setVertex(0, v);
        e->setMeasurement(measurement);
        e->setInformation(Eigen::Matrix2d::Identity());
        e->setId(i);
        optimizer.addEdge(e);
    }

    // Optimize step by step and log progress
    optimizer.initializeOptimization();
    int maxIterations = 20;
    for (int iter = 0; iter <= maxIterations; ++iter) {
        if (iter > 0) optimizer.optimize(1); // one iteration at a time
        
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
            << totalError << "\n";

        std::cout << "Iter " << iter 
                  << ": Position = (" << v->estimate().x() << ", " << v->estimate().y() 
                  << "), Error = " << totalError << std::endl;
    }
    csv.close();

    // Output final result
    std::cout << "Final optimized position: (" 
              << v->estimate().x() << ", " << v->estimate().y() << ")" << std::endl;
    std::cout << "True position: (" 
              << true_position.x() << ", " << true_position.y() << ")" << std::endl;

    return 0;
}

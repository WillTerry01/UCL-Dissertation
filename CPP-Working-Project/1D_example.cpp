#include <iostream>
#include <fstream>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <memory>

using namespace std;
using namespace g2o;

// Define a 1D vertex (state)
class Vertex1D : public BaseVertex<1, double> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = 0.0;
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += update[0];
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

// Unary edge representing a measurement of the 1D position
class UnaryEdge1D : public BaseUnaryEdge<1, double, Vertex1D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void computeError() override {
        const Vertex1D* v = static_cast<const Vertex1D*>(_vertices[0]);
        _error[0] = _measurement - v->estimate();
    }

    virtual bool read(std::istream&) override { return false; }
    virtual bool write(std::ostream&) const override { return false; }
};

int main() {
    // Setup the optimizer
    typedef BlockSolver< BlockSolverTraits<1, 1> > BlockSolverType;
    typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    );

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Add vertex (state)
    Vertex1D* v = new Vertex1D();
    v->setId(0);
    v->setEstimate(0.0);  // initial guess
    optimizer.addVertex(v);

    // Add measurements (e.g., z = x + noise)
    double measurements[] = {1.0, 1.1, 0.9, 1.05};
    for (int i = 0; i < 4; ++i) {
        UnaryEdge1D* e = new UnaryEdge1D();
        e->setVertex(0, v);
        e->setMeasurement(measurements[i]);
        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        e->setId(i);
        optimizer.addEdge(e);
    }

    // Open CSV file for writing progress
    std::ofstream csv("progress.csv");
    csv << "iteration,estimate,total_error\n";

    // Optimize step by step and log progress
    optimizer.initializeOptimization();
    int maxIterations = 20;
    for (int iter = 0; iter <= maxIterations; ++iter) {
        if (iter > 0) optimizer.optimize(1); // one iteration at a time
        // Compute total error
        double totalError = 0.0;
        for (const auto& edge : optimizer.edges()) {
            UnaryEdge1D* e = static_cast<UnaryEdge1D*>(edge);
            e->computeError();  // Make sure error is up to date
            totalError += e->error().squaredNorm();  // Sum of squared errors
        }
        csv << iter << "," << v->estimate() << "," << totalError << "\n";
        // Optional: ASCII bar
        std::cout << "Iter " << iter << ": Estimate = " << v->estimate() << ", Total Error = " << totalError << std::endl;
    }
    csv.close();

    // Output final result
    std::cout << "Optimized estimate: " << v->estimate() << std::endl;

    return 0;
}

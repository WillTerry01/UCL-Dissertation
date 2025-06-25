#include <iostream>
#include <fstream>
#include <vector>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <Eigen/Dense>
#include <random>
#include <memory>
#include <cmath>
#include <g2o/core/marginal_covariance_cholesky.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/optimizable_graph.h>

using namespace std;
using namespace g2o;

// Vertex: 2D position and velocity [x, y, vx, vy]
class Vertex4D : public BaseVertex<4, Eigen::Vector4d> {
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
class EdgeProcessModel : public BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex4D* v1 = static_cast<const Vertex4D*>(_vertices[0]);
        const Vertex4D* v2 = static_cast<const Vertex4D*>(_vertices[1]);
        // Predict v2 from v1 using constant velocity model
        double dt = 1.0; // time step
        Eigen::Vector4d pred;
        pred.head<2>() = v1->estimate().head<2>() + v1->estimate().tail<2>() * dt;
        pred.tail<2>() = v1->estimate().tail<2>();
        _error = v2->estimate() - pred;
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Measurement edge: connects state to observed position
class EdgeMeasurement : public BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex4D* v = static_cast<const Vertex4D*>(_vertices[0]);
        _error = _measurement - v->estimate().head<2>();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

int main() {
    // Parameters
    int N = 20; // number of time steps
    Eigen::Vector2d true_position(0.0, 0.0);
    Eigen::Vector2d true_velocity(1.0, 1.0);
    double dt = 1.0;
    Eigen::Matrix4d Q = Eigen::Matrix4d::Identity() * 0.05; // process noise
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 0.5;  // measurement noise

    // Ask user for trajectory type
    int traj_type = 1;
    std::cout << "Select trajectory type (1 = line, 2 = circle, 3 = square): ";
    std::cin >> traj_type;
    if (traj_type < 1 || traj_type > 3) {
        std::cout << "Invalid input. Using line trajectory (1)." << std::endl;
        traj_type = 1;
    }

    // Simulate true trajectory and measurements
    std::vector<Eigen::Vector4d> true_states(N);
    std::vector<Eigen::Vector2d> measurements(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_p(0.0, sqrt(R(0,0)));
    std::normal_distribution<> noise_q(0.0, sqrt(Q(0,0)));

    true_states[0] << true_position.x(), true_position.y(), true_velocity.x(), true_velocity.y();
    measurements[0] = true_states[0].head<2>() + Eigen::Vector2d(noise_p(gen), noise_p(gen));
    double omega = 0.2; // angular velocity (radians per time step)
    double square_side = 5.0; // length of square side
    int square_steps = N / 4; // steps per side
    for (int k = 1; k < N; ++k) {
        Eigen::Vector4d process_noise;
        process_noise << noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen);

        if (traj_type == 1) {
            // Line: constant velocity
            true_states[k].head<2>() = true_states[k-1].head<2>() + true_states[k-1].tail<2>() * dt;
            true_states[k].tail<2>() = true_states[k-1].tail<2>();
        } else if (traj_type == 2) {
            // Circle: rotate velocity
            double theta = omega * dt;
            Eigen::Matrix2d Rmat;
            Rmat << cos(theta), -sin(theta),
                    sin(theta),  cos(theta);
            Eigen::Vector2d prev_vel = true_states[k-1].tail<2>();
            Eigen::Vector2d new_vel = Rmat * prev_vel;
            true_states[k].head<2>() = true_states[k-1].head<2>() + new_vel * dt;
            true_states[k].tail<2>() = new_vel;
        } else if (traj_type == 3) {
            // Square: change velocity direction every square_steps
            int side = (k / square_steps) % 4;
            Eigen::Vector2d vel;
            if (side == 0)      vel = Eigen::Vector2d(1.0, 0.0); // right
            else if (side == 1) vel = Eigen::Vector2d(0.0, 1.0); // up
            else if (side == 2) vel = Eigen::Vector2d(-1.0, 0.0); // left
            else                vel = Eigen::Vector2d(0.0, -1.0); // down
            true_states[k].head<2>() = true_states[k-1].head<2>() + vel * dt;
            true_states[k].tail<2>() = vel;
        }
        // Add process noise
        true_states[k] += process_noise;
        // Measurement
        measurements[k] = true_states[k].head<2>() + Eigen::Vector2d(noise_p(gen), noise_p(gen));
    }

    // Print the actual (true) trajectory
    std::cout << "True trajectory (x, y):" << std::endl;
    for (int k = 0; k < N; ++k) {
        std::cout << k << ": (" << true_states[k][0] << ", " << true_states[k][1] << ")" << std::endl;
    }

    // Set up optimizer
    typedef BlockSolver< BlockSolverTraits<4, 4> > BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));
    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Create vertices
    std::vector<Vertex4D*> vertices(N);
    for (int k = 0; k < N; ++k) {
        vertices[k] = new Vertex4D();
        vertices[k]->setId(k);
        // Initial guess: noisy version of true state
        vertices[k]->setEstimate(true_states[k] + Eigen::Vector4d(noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen)));
        optimizer.addVertex(vertices[k]);
    }

    // Add process model edges
    for (int k = 1; k < N; ++k) {
        EdgeProcessModel* e = new EdgeProcessModel();
        e->setVertex(0, vertices[k-1]);
        e->setVertex(1, vertices[k]);
        e->setMeasurement(Eigen::Vector4d::Zero()); // Not used
        e->setInformation(Q.inverse());
        optimizer.addEdge(e);
    }

    // Add measurement edges
    for (int k = 0; k < N; ++k) {
        EdgeMeasurement* e = new EdgeMeasurement();
        e->setVertex(0, vertices[k]);
        e->setMeasurement(measurements[k]);
        e->setInformation(R.inverse());
        optimizer.addEdge(e);
    }

    // Optimize
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // Output results to CSV
    std::ofstream csv("../2d_trajectory_estimate.csv");
    csv << "t,true_x,true_y,meas_x,meas_y,est_x,est_y\n";
    for (int k = 0; k < N; ++k) {
        const auto& true_state = true_states[k];
        const auto& meas = measurements[k];
        const auto& v = vertices[k]->estimate();
        csv << k << ","
            << true_state[0] << "," << true_state[1] << ","
            << meas[0] << "," << meas[1] << ","
            << v[0] << "," << v[1] << "\n";
    }
    // Append chi2 value at the end
    csv << "chi2," << optimizer.chi2() << "\n";
    csv.close();

    // Print final chi2 value
    std::cout << "Final chi2: " << optimizer.chi2() << std::endl;

    return 0;
} 
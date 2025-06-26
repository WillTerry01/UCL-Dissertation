#include "1D_factor_graph_trajectory.h"
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>

using namespace std;
using namespace g2o;

FactorGraph1DTrajectory::FactorGraph1DTrajectory() : N_(0), chi2_(0.0) {
    Q_ = Eigen::Matrix2d::Identity() * 0.05;
    R_ = Eigen::Matrix<double, 1, 1>::Identity() * 0.5;
}

void FactorGraph1DTrajectory::run(const std::vector<Eigen::Vector2d>& true_states, const std::vector<double>* measurements, bool add_process_noise) {
    N_ = static_cast<int>(true_states.size());
    true_states_ = true_states;
    if (add_process_noise && N_ > 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise_q(0.0, sqrt(Q_(0,0)));
        for (int k = 1; k < N_; ++k) {
            Eigen::Vector2d process_noise;
            process_noise << noise_q(gen), noise_q(gen);
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
            measurements_[k] = true_states_[k][0] + noise_p(gen);
        }
    }
    
    setupOptimizer();
    optimize();
}

void FactorGraph1DTrajectory::setupOptimizer() {
    typedef BlockSolver< BlockSolverTraits<2, 2> > BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer_ = std::make_unique<SparseOptimizer>();
    optimizer_->setAlgorithm(solver);
    optimizer_->setVerbose(false);

    vertices_.resize(N_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_q(0.0, sqrt(Q_(0,0)));
    for (int k = 0; k < N_; ++k) {
        vertices_[k] = new Vertex2D();
        vertices_[k]->setId(k);
        vertices_[k]->setEstimate(true_states_[k] + Eigen::Vector2d(10*noise_q(gen), 10*noise_q(gen)));
        optimizer_->addVertex(vertices_[k]);
    }
    for (int k = 1; k < N_; ++k) {
        EdgeProcessModel1D* e = new EdgeProcessModel1D();
        e->setVertex(0, vertices_[k-1]);
        e->setVertex(1, vertices_[k]);
        e->setMeasurement(Eigen::Vector2d::Zero());
        e->setInformation(Q_.inverse());
        optimizer_->addEdge(e);
    }
    for (int k = 0; k < N_; ++k) {
        EdgeMeasurement1D* e = new EdgeMeasurement1D();
        e->setVertex(0, vertices_[k]);
        e->setMeasurement(measurements_[k]);
        e->setInformation(R_.inverse());
        optimizer_->addEdge(e);
    }
}

void FactorGraph1DTrajectory::optimize() {
    optimizer_->initializeOptimization();
    optimizer_->optimize(20);
    chi2_ = optimizer_->chi2();
}

double FactorGraph1DTrajectory::getChi2() const {
    return chi2_;
}

void FactorGraph1DTrajectory::writeCSV(const std::string& filename) const {
    std::ofstream csv(filename);
    csv << "t,true_x,meas_x,est_x\n";
    for (int k = 0; k < N_; ++k) {
        const auto& true_state = true_states_[k];
        double meas = measurements_[k];
        const auto& v = vertices_[k]->estimate();
        csv << k << ","
            << true_state[0] << ","
            << meas << ","
            << v[0] << "\n";
    }
    csv << "chi2," << chi2_ << "\n";
    csv.close();
} 
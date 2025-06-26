#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <cmath>

using namespace std;
using namespace g2o;

FactorGraph2DTrajectory::FactorGraph2DTrajectory() : N_(0), chi2_(0.0) {
    Q_ = Eigen::Matrix4d::Identity() * 0.1;
    R_ = Eigen::Matrix2d::Identity() * 0.5;
}

void FactorGraph2DTrajectory::run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, bool add_process_noise) {
    N_ = static_cast<int>(true_states.size());
    true_states_ = true_states;
    if (add_process_noise && N_ > 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> noise_q(0.0, sqrt(Q_(0,0)));
        for (int k = 1; k < N_; ++k) {
            Eigen::Vector4d process_noise;
            process_noise << noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen);
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
            measurements_[k] = true_states_[k].head<2>() + Eigen::Vector2d(noise_p(gen), noise_p(gen));
        }
    }

    // //print the q and r parameters
    // std::cout << "Q: " << Q_ << std::endl;
    // std::cout << "R: " << R_ << std::endl;ers
    // std::cout << "Q: " << Q_ << std::endl;
    // std::cout << "R: " << R_ << std::e

    setupOptimizer();
    optimize();
}

void FactorGraph2DTrajectory::setupOptimizer() {
    typedef BlockSolver< BlockSolverTraits<4, 4> > BlockSolverType;
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
        vertices_[k] = new Vertex4D();
        vertices_[k]->setId(k);
        vertices_[k]->setEstimate(true_states_[k] + Eigen::Vector4d(noise_q(gen), noise_q(gen), noise_q(gen), noise_q(gen)));
        optimizer_->addVertex(vertices_[k]);
    }
    for (int k = 1; k < N_; ++k) {
        EdgeProcessModel* e = new EdgeProcessModel();
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
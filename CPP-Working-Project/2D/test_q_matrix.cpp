#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <random>

int main() {
    // Load parameters from YAML file
    YAML::Node config = YAML::LoadFile("../BO_Parameters.yaml");
    // Test parameters
    double q = config["Data_Generation"]["q"].as<double>();  // Continuous-time white noise intensity
    double dt = config["Data_Generation"]["dt"].as<double>();  // Time step
    
    // Construct the process noise covariance matrix Q for 2D linear tracking
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    
    // Position-position covariance (diagonal)
    Q(0, 0) = q * dt4 / 4.0;  // x position variance
    Q(1, 1) = q * dt4 / 4.0;  // y position variance
    
    // Velocity-velocity covariance (diagonal)
    Q(2, 2) = q * dt2;        // x velocity variance
    Q(3, 3) = q * dt2;        // y velocity variance
    
    // Position-velocity cross covariance
    Q(0, 2) = q * dt3 / 2.0;  // x position - x velocity covariance
    Q(2, 0) = Q(0, 2);        // symmetric
    Q(1, 3) = q * dt3 / 2.0;  // y position - y velocity covariance
    Q(3, 1) = Q(1, 3);        // symmetric
    
    std::cout << "Q Matrix for 2D Linear Tracking Model:" << std::endl;
    std::cout << "q = " << q << ", dt = " << dt << std::endl;
    std::cout << std::fixed << std::setprecision(6) << Q << std::endl;
    
    // Verify symmetry
    std::cout << "\nQ is symmetric: " << (Q.isApprox(Q.transpose()) ? "YES" : "NO") << std::endl;
    
    // Test Cholesky decomposition
    Eigen::LLT<Eigen::Matrix4d> lltOfQ(Q);
    bool cholesky_success = (lltOfQ.info() == Eigen::Success);
    std::cout << "Cholesky decomposition successful: " << (cholesky_success ? "YES" : "NO") << std::endl;
    
    // Test LDLT decomposition (more robust)
    Eigen::LDLT<Eigen::Matrix4d> ldltOfQ(Q);
    bool ldlt_success = (ldltOfQ.info() == Eigen::Success);
    std::cout << "LDLT decomposition successful: " << (ldlt_success ? "YES" : "NO") << std::endl;
    
    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(Q);
    Eigen::Vector4d eigenvalues = eigensolver.eigenvalues();
    std::cout << "\nEigenvalues: " << eigenvalues.transpose() << std::endl;
    
    // Check condition number
    double condition_number = eigenvalues.maxCoeff() / eigenvalues.minCoeff();
    std::cout << "Condition number: " << condition_number << std::endl;
    
    // Print individual elements for verification
    std::cout << "\nIndividual elements:" << std::endl;
    std::cout << "Q(0,0) = q*dt^4/4 = " << q << "*" << dt4 << "/4 = " << Q(0,0) << std::endl;
    std::cout << "Q(2,2) = q*dt^2 = " << q << "*" << dt2 << " = " << Q(2,2) << std::endl;
    std::cout << "Q(0,2) = q*dt^3/2 = " << q << "*" << dt3 << "/2 = " << Q(0,2) << std::endl;
    
    // Test noise generation
    std::cout << "\n=== Testing Noise Generation ===" << std::endl;
    std::mt19937 gen(42);
    std::normal_distribution<> normal_dist(0.0, 1.0);
    
    if (cholesky_success) {
        std::cout << "Using Cholesky decomposition for noise generation" << std::endl;
        Eigen::Matrix4d L = lltOfQ.matrixL();
        
        // Generate sample noise
        Eigen::Vector4d uncorrelated_noise;
        for (int i = 0; i < 4; ++i) {
            uncorrelated_noise[i] = normal_dist(gen);
        }
        Eigen::Vector4d correlated_noise = L * uncorrelated_noise;
        std::cout << "Sample correlated noise: " << correlated_noise.transpose() << std::endl;
    } else if (ldlt_success) {
        std::cout << "Using LDLT decomposition for noise generation" << std::endl;
        Eigen::Vector4d eigenvalues_ldlt = ldltOfQ.vectorD();
        Eigen::Matrix4d eigenvectors_ldlt = ldltOfQ.matrixL();
        
        // Generate sample noise
        Eigen::Vector4d uncorrelated_noise;
        for (int i = 0; i < 4; ++i) {
            if (std::abs(eigenvalues_ldlt[i]) > 1e-12) {
                uncorrelated_noise[i] = normal_dist(gen) * std::sqrt(std::abs(eigenvalues_ldlt[i]));
            } else {
                uncorrelated_noise[i] = 0.0;
            }
        }
        Eigen::Vector4d correlated_noise = eigenvectors_ldlt * uncorrelated_noise;
        std::cout << "Sample correlated noise: " << correlated_noise.transpose() << std::endl;
    } else {
        std::cout << "WARNING: Both decompositions failed!" << std::endl;
    }
    
    return 0;
} 
#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <iomanip>

int main() {
    // Test the factor graph Q matrix construction
    FactorGraph2DTrajectory fg;
    
    // Test with different q values
    std::vector<double> q_values = {0.1, 0.25, 1.0, 2.0};
    
    for (double q : q_values) {
        std::cout << "\n=== Testing with q = " << q << " ===" << std::endl;
        
        // Set Q using the proper method
        fg.setQFromProcessNoiseIntensity(q, 1.0);
        
        std::cout << "Q Matrix:" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << fg.Q_ << std::endl;
        
        // Verify symmetry
        std::cout << "Q is symmetric: " << (fg.Q_.isApprox(fg.Q_.transpose()) ? "YES" : "NO") << std::endl;
        
        // Test Cholesky decomposition
        Eigen::LLT<Eigen::Matrix4d> lltOfQ(fg.Q_);
        bool cholesky_success = (lltOfQ.info() == Eigen::Success);
        std::cout << "Cholesky decomposition successful: " << (cholesky_success ? "YES" : "NO") << std::endl;
        
        // Test LDLT decomposition
        Eigen::LDLT<Eigen::Matrix4d> ldltOfQ(fg.Q_);
        bool ldlt_success = (ldltOfQ.info() == Eigen::Success);
        std::cout << "LDLT decomposition successful: " << (ldlt_success ? "YES" : "NO") << std::endl;
        
        // Check eigenvalues
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(fg.Q_);
        Eigen::Vector4d eigenvalues = eigensolver.eigenvalues();
        std::cout << "Eigenvalues: " << eigenvalues.transpose() << std::endl;
        
        // Print individual elements for verification
        std::cout << "Q(0,0) = q*dt^4/4 = " << q << "*1^4/4 = " << q/4.0 << std::endl;
        std::cout << "Q(2,2) = q*dt^2 = " << q << "*1^2 = " << q << std::endl;
        std::cout << "Q(0,2) = q*dt^3/2 = " << q << "*1^3/2 = " << q/2.0 << std::endl;
    }
    
    // Test the legacy scalar method for comparison
    std::cout << "\n=== Testing legacy scalar method ===" << std::endl;
    fg.setQFromScalar(0.25, 1.0);
    std::cout << "Q Matrix (scalar):" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << fg.Q_ << std::endl;
    
    return 0;
} 
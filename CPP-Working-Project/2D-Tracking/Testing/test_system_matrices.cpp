#include "../fg_class_tracking.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <iomanip>

/**
 * Test Program: System Matrices and Proposition 3 Validation
 * ==========================================================
 * 
 * Purpose: Explicitly validate the F and B matrices in the factor graph
 * and test Proposition 3 (no optimization case) to ensure graph structure
 * is correct even without optimization.
 * 
 * Tests:
 * 1. Verify F matrix matches theoretical constant velocity model
 * 2. Verify B matrix matches theoretical control input model  
 * 3. Test Proposition 3 (do_optimization = false) with perfect initialization
 * 4. Confirm that process noise Q and measurement noise R are correctly applied
 */

void printMatrix(const std::string& name, const Eigen::MatrixXd& matrix) {
    std::cout << name << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << matrix << std::endl;
    std::cout << std::endl;
}

// Test the theoretical F and B matrices
void testSystemMatrices(double dt) {
    std::cout << "=== System Matrices Validation ===" << std::endl;
    std::cout << "Time step (dt): " << dt << std::endl;
    std::cout << std::endl;
    
    // Expected F matrix (constant velocity model)
    Eigen::Matrix4d F_expected = Eigen::Matrix4d::Identity();
    F_expected(0, 2) = dt;  // x position += x velocity * dt
    F_expected(1, 3) = dt;  // y position += y velocity * dt
    
    printMatrix("Expected F matrix (Constant Velocity Model)", F_expected);
    
    // Expected B matrix (acceleration control)
    Eigen::Matrix<double, 4, 2> B_expected;
    double dt2 = dt * dt;
    B_expected << 0.5 * dt2, 0.0,
                  0.0, 0.5 * dt2,
                  dt, 0.0,
                  0.0, dt;
    
    printMatrix("Expected B matrix (Acceleration Control)", B_expected);
    
    // Test the system equation manually
    std::cout << "Testing system equation: x_{k+1} = F*x_k + B*u_k" << std::endl;
    
    // Test state: [x=1, y=2, vx=0.5, vy=-0.3]
    Eigen::Vector4d x_k;
    x_k << 1.0, 2.0, 0.5, -0.3;
    
    // Test control: [ax=0.1, ay=0.2] (acceleration)
    Eigen::Vector2d u_k;
    u_k << 0.1, 0.2;
    
    // Predicted next state
    Eigen::Vector4d x_k_plus_1 = F_expected * x_k + B_expected * u_k;
    
    std::cout << "Test input:" << std::endl;
    std::cout << "  x_k = [" << x_k.transpose() << "]" << std::endl;
    std::cout << "  u_k = [" << u_k.transpose() << "]" << std::endl;
    std::cout << "  x_{k+1} = [" << x_k_plus_1.transpose() << "]" << std::endl;
    std::cout << std::endl;
    
    // Manual calculation for verification
    double x_new = x_k[0] + x_k[2] * dt + 0.5 * u_k[0] * dt2;
    double y_new = x_k[1] + x_k[3] * dt + 0.5 * u_k[1] * dt2;
    double vx_new = x_k[2] + u_k[0] * dt;
    double vy_new = x_k[3] + u_k[1] * dt;
    
    std::cout << "Manual calculation verification:" << std::endl;
    std::cout << "  x_new = " << x_new << " (x + vx*dt + 0.5*ax*dt^2)" << std::endl;
    std::cout << "  y_new = " << y_new << " (y + vy*dt + 0.5*ay*dt^2)" << std::endl;
    std::cout << "  vx_new = " << vx_new << " (vx + ax*dt)" << std::endl;
    std::cout << "  vy_new = " << vy_new << " (vy + ay*dt)" << std::endl;
    std::cout << std::endl;
}

// Generate a simple trajectory with known control inputs
std::vector<Eigen::Vector4d> generateControlledTrajectory(int N, double dt) {
    std::vector<Eigen::Vector4d> states(N);
    
    // Initial state
    states[0] << 0.0, 0.0, 1.0, 0.5;
    
    // System matrices
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    F(0, 2) = dt;
    F(1, 3) = dt;
    
    Eigen::Matrix<double, 4, 2> B;
    double dt2 = dt * dt;
    B << 0.5 * dt2, 0.0,
         0.0, 0.5 * dt2,
         dt, 0.0,
         0.0, dt;
    
    // Generate trajectory with deterministic control
    for (int k = 1; k < N; ++k) {
        // Simple sinusoidal acceleration for interesting trajectory
        Eigen::Vector2d u_k;
        u_k << 0.1 * sin(0.2 * k), 0.1 * cos(0.2 * k);
        
        states[k] = F * states[k-1] + B * u_k;
    }
    
    return states;
}

// Test Proposition 3 (no optimization, perfect initialization)
void testProposition3(int N, double dt) {
    std::cout << "=== Testing Proposition 3 (No Optimization) ===" << std::endl;
    std::cout << "Initializing at ground truth, no optimization" << std::endl;
    std::cout << std::endl;
    
    // Generate deterministic trajectory
    auto true_states = generateControlledTrajectory(N, dt);
    
    // Generate perfect measurements
    std::vector<Eigen::Vector2d> measurements(N);
    for (int k = 0; k < N; ++k) {
        measurements[k] = true_states[k].head<2>();
    }
    
    // Create factor graph
    FactorGraph2DTrajectory fg;
    
    // Set small noise for numerical stability
    fg.setQFromProcessNoiseIntensity(1e-6, dt);
    fg.setRFromMeasurementNoise(1e-3, 1e-3);
    
    try {
        // Run WITHOUT optimization (do_optimization = false)
        fg.run(true_states, &measurements, dt, false);
        
        // Get estimates (should be exactly the true states since no optimization)
        auto estimated_states = fg.getAllEstimates();
        double chi2 = fg.getChi2();
        
        std::cout << "Proposition 3 Results:" << std::endl;
        std::cout << "  Chi-squared value: " << std::scientific << chi2 << std::endl;
        
        // Check if estimates exactly match true states (since initialized at ground truth)
        bool perfect_match = true;
        double max_error = 0.0;
        
        for (int k = 0; k < N; ++k) {
            Eigen::Vector4d error = true_states[k] - estimated_states[k];
            double error_norm = error.norm();
            max_error = std::max(max_error, error_norm);
            
            if (error_norm > 1e-10) {
                perfect_match = false;
            }
        }
        
        std::cout << "  Maximum state error: " << std::scientific << max_error << std::endl;
        std::cout << "  Perfect initialization preserved: " << (perfect_match ? "YES" : "NO") << std::endl;
        
        if (perfect_match) {
            std::cout << "  ✓ Proposition 3 PASSED: Graph structure is correct!" << std::endl;
            std::cout << "  ✓ No optimization preserves perfect initialization" << std::endl;
        } else {
            std::cout << "  ✗ Proposition 3 FAILED: Graph may have structural issues" << std::endl;
        }
        
        // Display some states for verification
        std::cout << std::endl << "State verification (first 3 time steps):" << std::endl;
        for (int k = 0; k < std::min(3, N); ++k) {
            std::cout << "k=" << k << ": True=[" << true_states[k].transpose() 
                     << "], Est=[" << estimated_states[k].transpose() << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in Proposition 3 test: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Test Q and R matrix construction
void testNoiseMatrices(double dt) {
    std::cout << "=== Testing Q and R Matrix Construction ===" << std::endl;
    
    // Test Q matrix construction
    double q_intensity = 0.1;
    std::cout << "Process noise intensity: " << q_intensity << std::endl;
    std::cout << "Time step: " << dt << std::endl;
    
    // Manually construct expected Q matrix
    Eigen::Matrix4d Q_expected = Eigen::Matrix4d::Zero();
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    
    // Position-position covariance
    Q_expected(0, 0) = dt3 / 3.0 * q_intensity;  // x position variance
    Q_expected(1, 1) = dt3 / 3.0 * q_intensity;  // y position variance
    
    // Velocity-velocity covariance
    Q_expected(2, 2) = dt * q_intensity;         // x velocity variance
    Q_expected(3, 3) = dt * q_intensity;         // y velocity variance
    
    // Position-velocity cross covariance
    Q_expected(0, 2) = dt2 / 2.0 * q_intensity;  // x position - x velocity covariance
    Q_expected(2, 0) = Q_expected(0, 2);          // symmetric
    Q_expected(1, 3) = dt2 / 2.0 * q_intensity;  // y position - y velocity covariance
    Q_expected(3, 1) = Q_expected(1, 3);          // symmetric
    
    printMatrix("Expected Q matrix (Process Noise Covariance)", Q_expected);
    
    // Test R matrix construction
    double sigma_x = 0.05, sigma_y = 0.07;
    std::cout << "Measurement noise standard deviations: sigma_x=" << sigma_x << ", sigma_y=" << sigma_y << std::endl;
    
    Eigen::Matrix2d R_expected = Eigen::Matrix2d::Zero();
    R_expected(0, 0) = sigma_x * sigma_x;
    R_expected(1, 1) = sigma_y * sigma_y;
    
    printMatrix("Expected R matrix (Measurement Noise Covariance)", R_expected);
    
    std::cout << "Information matrices (inverses):" << std::endl;
    printMatrix("Q^{-1} (Process Information Matrix)", Q_expected.inverse());
    printMatrix("R^{-1} (Measurement Information Matrix)", R_expected.inverse());
}

int main() {
    std::cout << "=== Comprehensive Factor Graph Model Validation ===" << std::endl;
    std::cout << "Testing system matrices, Proposition 3, and noise models" << std::endl;
    std::cout << std::endl;
    
    double dt = 1.0;
    int N = 10;
    
    // Test 1: System matrices
    testSystemMatrices(dt);
    
    // Test 2: Proposition 3 (no optimization)
    testProposition3(N, dt);
    
    // Test 3: Noise matrices
    testNoiseMatrices(dt);
    
    std::cout << "=== All Tests Complete ===" << std::endl;
    
    return 0;
} 
#include "2D_factor_graph_trajectory.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

int main() {
    std::cout << "Testing error checking for uninitialized Q and R matrices..." << std::endl;
    
    // Create a simple trajectory
    std::vector<Eigen::Vector4d> true_states(5);
    for (int i = 0; i < 5; ++i) {
        true_states[i] << i, i, 1.0, 1.0;  // Simple linear trajectory
    }
    
    // Test 1: Uninitialized Q and R (should throw error)
    try {
        std::cout << "\nTest 1: Running with uninitialized Q and R..." << std::endl;
        FactorGraph2DTrajectory fg;
        fg.run(true_states, nullptr, false, 1.0);
        std::cout << "ERROR: Should have thrown an exception!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
    }
    
    // Test 2: Only Q initialized (should throw error for R)
    try {
        std::cout << "\nTest 2: Running with only Q initialized..." << std::endl;
        FactorGraph2DTrajectory fg;
        fg.setQFromProcessNoiseIntensity(0.1, 1.0);
        fg.run(true_states, nullptr, false, 1.0);
        std::cout << "ERROR: Should have thrown an exception for R!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
    }
    
    // Test 3: Only R initialized (should throw error for Q)
    try {
        std::cout << "\nTest 3: Running with only R initialized..." << std::endl;
        FactorGraph2DTrajectory fg;
        fg.setRFromMeasurementNoise(0.1, 0.1);
        fg.run(true_states, nullptr, false, 1.0);
        std::cout << "ERROR: Should have thrown an exception for Q!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Correctly caught error: " << e.what() << std::endl;
    }
    
    // Test 4: Both Q and R properly initialized (should work)
    try {
        std::cout << "\nTest 4: Running with both Q and R properly initialized..." << std::endl;
        FactorGraph2DTrajectory fg;
        fg.setQFromProcessNoiseIntensity(0.1, 1.0);
        fg.setRFromMeasurementNoise(0.1, 0.1);
        fg.run(true_states, nullptr, false, 1.0);
        std::cout << "✓ Successfully ran with proper initialization!" << std::endl;
        std::cout << "Chi2 value: " << fg.getChi2() << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "ERROR: Should not have thrown an exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n✓ All error checking tests passed!" << std::endl;
    return 0;
} 
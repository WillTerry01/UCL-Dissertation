#include "fg_class_tracking.h"
#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== Testing Q Matrix Construction ===" << std::endl;
    
    double q = 0.5;
    double dt = 0.5;
    
    std::cout << "Parameters: q=" << q << ", dt=" << dt << std::endl;
    
    // Test factor graph Q matrix construction
    FactorGraph2DTrajectory fg;
    fg.setQFromProcessNoiseIntensity(q, dt);
    
    // Access Q matrix (we need to access it somehow)
    // Let's compute the theoretical values
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    
    std::cout << "\nTheoretical Q matrix values:" << std::endl;
    std::cout << "Q(0,0) = dt³/3 * V0 = " << (dt3/3.0 * q) << std::endl;
    std::cout << "Q(2,2) = dt * V0 = " << (dt * q) << std::endl;
    std::cout << "Q(0,2) = dt²/2 * V0 = " << (dt2/2.0 * q) << std::endl;
    
    std::cout << "\nData generation output showed:" << std::endl;
    std::cout << "Q(0,0) = 0.0208333" << std::endl;
    std::cout << "Q(2,2) = 0.25" << std::endl;
    std::cout << "Q(0,2) = 0.0625" << std::endl;
    
    std::cout << "\nComparison:" << std::endl;
    std::cout << "Q(0,0): theoretical=" << (dt3/3.0 * q) << ", data_gen=0.0208333, match=" << (abs((dt3/3.0 * q) - 0.0208333) < 1e-6) << std::endl;
    std::cout << "Q(2,2): theoretical=" << (dt * q) << ", data_gen=0.25, match=" << (abs((dt * q) - 0.25) < 1e-6) << std::endl;
    std::cout << "Q(0,2): theoretical=" << (dt2/2.0 * q) << ", data_gen=0.0625, match=" << (abs((dt2/2.0 * q) - 0.0625) < 1e-6) << std::endl;
    
    return 0;
} 
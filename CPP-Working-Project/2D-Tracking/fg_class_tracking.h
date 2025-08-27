#pragma once
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/core/optimizable_graph.h>
#include <H5Cpp.h>

// Vertex: 2D position and velocity [x, y, vx, vy]
class Vertex4D : public g2o::BaseVertex<4, Eigen::Vector4d> {
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
class EdgeProcessModel : public g2o::BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructor to set dt and control input
    EdgeProcessModel(double dt = 1.0) : dt_(dt) {}
    
    void computeError() override {
        const Vertex4D* v1 = static_cast<const Vertex4D*>(_vertices[0]);
        const Vertex4D* v2 = static_cast<const Vertex4D*>(_vertices[1]);
        
        // State transition matrix F (constant velocity model)
        // F = [1  0  dt  0]
        //     [0  1  0   dt]
        //     [0  0  1   0]
        //     [0  0  0   1]
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0, 2) = dt_;  // x position += x velocity * dt
        F(1, 3) = dt_;  // y position += y velocity * dt
        
        // Control input matrix B (for future acceleration control)
        // B = [0.5 * dt^2  0        ]
        //     [0           0.5 * dt^2]
        //     [dt          0        ]
        //     [0           dt       ]
        Eigen::Matrix<double, 4, 2> B;
        double dt2 = dt_ * dt_;
        B << 0.5 * dt2, 0.0,
             0.0, 0.5 * dt2,
             dt_, 0.0,
             0.0, dt_;
        
        // Get control input (acceleration) - set to zero for constant velocity
        // TODO: Modify this to get actual control input for controlled motion
        Eigen::Vector2d acceleration = getControlInput();  // Currently returns zero
        
        // State equation: xₖ₊₁ = Fxₖ + Buₖ + vₖ
        Eigen::Vector4d control_effect = B * acceleration;
        Eigen::Vector4d pred = F * v1->estimate() + control_effect;
        
        // Error: e = xₖ₊₁ - (Fxₖ + Buₖ)
        _error = v2->estimate() - pred;
    }
    
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    
private:
    double dt_;
    
    // Control input generation function - currently returns zero for constant velocity
    // TODO: Modify this function to return actual control input for controlled motion
    Eigen::Vector2d getControlInput() {
        // For now, return zero acceleration (constant velocity)
        // This can be modified later to return:
        // - Sinusoidal acceleration: [A*sin(ω*t), A*cos(ω*t)]
        // - Step acceleration: [a_x, a_y] for t > t_switch
        // - Random acceleration: [N(0,σ²), N(0,σ²)]
        // - Nonlinear control laws
        return Eigen::Vector2d::Zero();
    }
};

// Measurement edge: connects state to observed position
class EdgeMeasurement : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex4D* v = static_cast<const Vertex4D*>(_vertices[0]);
        _error = _measurement - v->estimate().head<2>();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

// Nonlinear motion model edge: constant turn rate model
class EdgeNonlinearMotionCT : public g2o::BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructor to set dt and turn rate
    EdgeNonlinearMotionCT(double dt = 1.0, double turn_rate = 0.0) : dt_(dt), turn_rate_(turn_rate) {}
    
    void computeError() override {
        const Vertex4D* v1 = static_cast<const Vertex4D*>(_vertices[0]);
        const Vertex4D* v2 = static_cast<const Vertex4D*>(_vertices[1]);
        
        // Nonlinear constant turn rate prediction
        Eigen::Vector4d pred = predictState(v1->estimate());
        
        // Error: e = xₖ₊₁ - f(xₖ, uₖ)
        _error = v2->estimate() - pred;
    }
    
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
    
private:
    double dt_;
    double turn_rate_;
    
    // Nonlinear state prediction for constant turn rate model
    Eigen::Vector4d predictState(const Eigen::Vector4d& x) const {
        double x_pos = x[0], y_pos = x[1], vx = x[2], vy = x[3];
        double v = std::sqrt(vx*vx + vy*vy);  // Speed
        double heading = std::atan2(vy, vx);  // Current heading
        
        if (std::abs(turn_rate_) < 1e-6) {
            // Straight line motion (constant velocity)
            return Eigen::Vector4d(x_pos + vx * dt_, y_pos + vy * dt_, vx, vy);
        } else {
            // Constant turn rate motion
            double new_heading = heading + turn_rate_ * dt_;
            double new_vx = v * std::cos(new_heading);
            double new_vy = v * std::sin(new_heading);
            
            // Position update (integrate velocity)
            double new_x_pos = x_pos + (v / turn_rate_) * (std::sin(new_heading) - std::sin(heading));
            double new_y_pos = y_pos - (v / turn_rate_) * (std::cos(new_heading) - std::cos(heading));
            
            return Eigen::Vector4d(new_x_pos, new_y_pos, new_vx, new_vy);
        }
    }
};

// GPS measurement edge: Cartesian position measurements (x, y) - same for both linear and nonlinear
class EdgeGPSMeasurement : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const Vertex4D* v = static_cast<const Vertex4D*>(_vertices[0]);
        // GPS measurement: direct position [x, y] - same for both linear and nonlinear systems
        _error = _measurement - v->estimate().head<2>();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

class FactorGraph2DTrajectory {
public:
    struct OutputOptions {
        bool output_estimated_state = false;
        bool output_true_state = false;
        bool output_information_matrix = false;
    };

    struct Chi2Breakdown {
        double processChi2 = 0.0;
        double measurementChi2 = 0.0;
        double totalChi2 = 0.0;
    };

    FactorGraph2DTrajectory();
    void setOutputOptions(const OutputOptions& options) { output_options_ = options; }
    OutputOptions getOutputOptions() const { return output_options_; }
    // Get all estimated states
    std::vector<Eigen::Vector4d> getAllEstimates() const { return getEstimatesInternal(); }
    // Get all true states
    std::vector<Eigen::Vector4d> getAllTrueStates() const { return true_states_; }
    // Get the full information matrix (Hessian)
    Eigen::MatrixXd getFullHessianMatrix() const;
    // Run the factor graph optimization with user-provided true states and optional measurements.
    // If do_optimization is false, graph is built but not optimized (for Proposition 3)
    void run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements = nullptr, double dt = 1.0, bool do_optimization = true);
    // Variable-dt overload (linear model)
    void run(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, const std::vector<double>& dt_vec, bool do_optimization = true);
    void writeCSV(const std::string& filename = "../H5_Files/2d_trajectory_estimate.csv") const;
    void writeHDF5(const std::string& filename = "../H5_Files/2d_trajectory_estimate.h5") const;
    double getChi2() const;
    // Getter for estimated state at time step k
    Eigen::Vector4d getEstimate(int k) const { return vertices_[k]->estimate(); }
    void printHessian() const;

    // Add these new methods
    void setQFromProcessNoiseIntensity(double q_intensity, double dt = 1.0);
    void setRFromMeasurementNoise(double sigma_x, double sigma_y);
    
    // New methods to calculate actual graph dimensions (like MATLAB student implementation)
    int getActualGraphDimZ() const;  // Sum of all edge dimensions
    int getActualGraphDimX() const;  // Sum of all vertex dimensions
    std::pair<int, int> getActualGraphDimensions() const;  // Returns {dimZ, dimX}
    
    // Nonlinear system methods
    void setMotionModelType(const std::string& model_type, double turn_rate = 0.0);
    void setMeasurementModelType(const std::string& model_type);  // Removed sensor_pos parameter
    void runNonlinear(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements = nullptr, double dt = 1.0, bool do_optimization = true);
    // Variable-dt overload (nonlinear model)
    void runNonlinear(const std::vector<Eigen::Vector4d>& true_states, const std::vector<Eigen::Vector2d>* measurements, const std::vector<double>& dt_vec, bool do_optimization = true);

    // Optimizer configuration and initialization controls
    void setMaxIterations(int max_iters) { max_iterations_ = max_iters; }
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setInitMode(const std::string& mode) { init_mode_ = mode; }
    void setInitJitter(double pos_std, double vel_std) { init_jitter_pos_std_ = pos_std; init_jitter_vel_std_ = vel_std; }
    Chi2Breakdown computeChi2Breakdown();

private:
    std::vector<Vertex4D*> vertices_;
    std::vector<Eigen::Vector4d> true_states_;
    std::vector<Eigen::Vector2d> measurements_;
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
    g2o::BlockSolver<g2o::BlockSolverTraits<4, 4>>* blockSolver_;
    int N_;
    double chi2_;
    Eigen::Matrix4d Q_;
    Eigen::Matrix2d R_;
    OutputOptions output_options_;
    double dt_;  // Add dt as a member variable
    double q_intensity_ = 0.0;  // Store process noise intensity for variable-dt edges
    
    // Nonlinear system parameters
    std::string motion_model_type_;  // "linear", "constant_turn_rate", "dubins"
    std::string measurement_model_type_;  // "linear", "gps" (both use Cartesian measurements)
    double turn_rate_;  // For constant turn rate model
    // Removed sensor_pos_ - not needed for GPS tracking
    
    // Helper method to extract optimized estimates from vertices
    std::vector<Eigen::Vector4d> getEstimatesInternal() const;
    void setupOptimizer();
    void optimize();

    // Helper to build Q(dt) from q_intensity
    Eigen::Matrix4d buildQ(double q_intensity, double dt) const;

    // Optimizer config and initialization
    int max_iterations_ = 100;
    bool verbose_ = false;
    std::string init_mode_ = "measurement"; // options: "zero", "measurement", "truth_plus_jitter"
    double init_jitter_pos_std_ = 0.1;
    double init_jitter_vel_std_ = 0.5;
}; 
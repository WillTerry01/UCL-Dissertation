#include <iostream>
#include <vector>
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>

// Define the objective function to optimize
class ExampleFunction : public bayesopt::ContinuousModel {
public:
    ExampleFunction(const bayesopt::Parameters& par) : ContinuousModel(2, par) {}

    double evaluateSample(const vectord& x) {
        // Example function: f(x,y) = -(x^2 + y^2) + 2*sin(x) + 2*sin(y)
        double x_val = x[0];
        double y_val = x[1];
        
        return -(x_val * x_val + y_val * y_val) + 
                2.0 * sin(x_val) + 2.0 * sin(y_val);
    }

    bool checkReachability(const vectord& query) {
        return (query[0] >= -5.0 && query[0] <= 5.0 &&
                query[1] >= -5.0 && query[1] <= 5.0);
    }
};

int main() {
    // Set up the parameters
    bayesopt::Parameters par = initialize_parameters_to_default();
    par.n_iterations = 50;
    par.random_seed = 0;
    par.verbose_level = 1;
    par.noise = 1e-10;

    // Create and run the optimization
    ExampleFunction opt(par);
    vectord result(2);

    opt.optimize(result);
    std::cout << "Result: " << result << " -> " 
              << opt.evaluateSample(result) << std::endl;

    return 0;
} 
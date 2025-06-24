cmake_minimum_required(VERSION 3.10)
project(BayesOptExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)

# Set BayesOpt paths manually since it's installed locally
set(BAYESOPT_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../bayesopt/include")
set(BAYESOPT_LIBRARIES 
    "${CMAKE_SOURCE_DIR}/../bayesopt/lib/libbayesopt.a"
    "${CMAKE_SOURCE_DIR}/../bayesopt/lib/libnlopt.a"
)

# Add the executable
add_executable(bayesopt_example bayesopt_example.cpp)

# Link against required libraries
target_link_libraries(bayesopt_example
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
)

# Include directories
target_include_directories(bayesopt_example
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(2d_factor_graph_trajectory 2D_factor_graph_trajectory.cpp)
target_link_libraries(2d_factor_graph_trajectory
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(2d_factor_graph_trajectory
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

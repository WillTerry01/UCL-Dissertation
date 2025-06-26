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

# Executable for single run demo
add_executable(2D_single_run 2D/2D_single_run.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_single_run
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(2D_single_run
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

# Executable for main application (100 runs)
add_executable(2D_main 2D/2D_main.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_main
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(2D_main
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

# Executable for generating noisy data for repeatable experiments
add_executable(2D_generate_noisy_data 2D/2D_generate_noisy_data.cpp)
target_link_libraries(2D_generate_noisy_data
    PRIVATE
    Eigen3::Eigen
)
target_include_directories(2D_generate_noisy_data
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

# Executable for running the factor graph on CSV-loaded data
add_executable(2D_main_from_csv 2D/2D_main_from_csv.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_main_from_csv
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(2D_main_from_csv
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

# Executable for 1D single run demo
add_executable(1D_single_run 1D/1D_single_run.cpp 1D/1D_factor_graph_trajectory.cpp)
target_link_libraries(1D_single_run
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(1D_single_run
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

# Executable for 1D main application (1000 runs)
add_executable(1D_main 1D/1D_main.cpp 1D/1D_factor_graph_trajectory.cpp)
target_link_libraries(1D_main
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(1D_main
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

# Executable for running BayesOpt on the factor graph using CSV-loaded data
add_executable(2D_bayesopt_from_csv 2D/2D_bayesopt_from_csv.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_bayesopt_from_csv
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
)
target_include_directories(2D_bayesopt_from_csv
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
)

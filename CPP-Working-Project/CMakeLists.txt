cmake_minimum_required(VERSION 3.10)
project(BayesOptExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module to fetch dependencies
include(FetchContent)

# Fetch yaml-cpp
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# Find required packages
find_package(Eigen3 REQUIRED)
# Set g2o_DIR to use the local build of g2o
set(g2o_DIR "/home/will/Dissertation/UCL-Dissertation/g2o/local_install/lib/cmake/g2o")
find_package(g2o REQUIRED)
find_package(OpenMP)
find_package(HDF5 REQUIRED COMPONENTS CXX)

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
    ${HDF5_LIBRARIES}
)
target_include_directories(2D_single_run
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
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
    ${HDF5_LIBRARIES}
)
target_include_directories(2D_main
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for generating noisy data for repeatable experiments
add_executable(2D_generate_noisy_data 2D/2D_generate_noisy_data.cpp)
target_link_libraries(2D_generate_noisy_data
    PRIVATE
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(2D_generate_noisy_data
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Executable for testing Q matrix construction
add_executable(test_q_matrix 2D/test_q_matrix.cpp)
target_link_libraries(test_q_matrix
    PRIVATE
    Eigen3::Eigen
    yaml-cpp
)
target_include_directories(test_q_matrix
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

# Executable for testing factor graph Q matrix
add_executable(test_factor_graph_q 2D/test_factor_graph_q.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(test_factor_graph_q
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
)
target_include_directories(test_factor_graph_q
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
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
    ${HDF5_LIBRARIES}
)
target_include_directories(2D_main_from_csv
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
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
    ${HDF5_LIBRARIES}
)
target_include_directories(1D_single_run
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
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
    ${HDF5_LIBRARIES}
)
target_include_directories(1D_main
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
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
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(2D_bayesopt_from_csv
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for single run Monte Carlo experiment (chi2 and MSE)
add_executable(2D_single_run_mc 2D/2D_single_run_mc.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_single_run_mc
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
)
target_include_directories(2D_single_run_mc
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)


# Executable for running BayesOpt on the factor graph using CSV-loaded data
add_executable(2D_bayesopt_from_csv_parallel 2D/2D_bayesopt_from_csv_parallel.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_bayesopt_from_csv_parallel
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(2D_bayesopt_from_csv_parallel
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

add_executable(2D_bayesopt_from_csv_parallel_CNEES 2D/2D_bayesopt_from_csv_parallel_CNEES.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_bayesopt_from_csv_parallel_CNEES
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(2D_bayesopt_from_csv_parallel_CNEES
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

add_executable(2D_bayesopt_NEES 2D/2D_bayesopt_NEES.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_bayesopt_NEES
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(2D_bayesopt_NEES
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for validating filter performance
add_executable(2D_validate_filter 2D/2D_validate_filter.cpp 2D/2D_factor_graph_trajectory.cpp)
target_link_libraries(2D_validate_filter
    PRIVATE
    ${BAYESOPT_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(2D_validate_filter
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)


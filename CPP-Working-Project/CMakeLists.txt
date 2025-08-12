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

# 2D-Tracking folder ##########################################################

# Executable for generating tracking data (clean trajectory + noisy measurements)
add_executable(tracking_gen_data_linear 2D-Tracking/tracking_gen_data_linear.cpp)
target_link_libraries(tracking_gen_data_linear
    PRIVATE
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(tracking_gen_data_linear
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Executable for generating nonlinear tracking data (constant turn rate + range-bearing)
add_executable(tracking_gen_data_nonlinear 2D-Tracking/tracking_gen_data_nonlinear.cpp)
target_link_libraries(tracking_gen_data_nonlinear
    PRIVATE
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(tracking_gen_data_nonlinear
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Executable for nonlinear factor graph tracking demo
add_executable(demo_nonlinear_tracking 2D-Tracking/demo_nonlinear_tracking.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(demo_nonlinear_tracking
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(demo_nonlinear_tracking
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for nonlinear Bayesian Optimization
add_executable(BO_Tracking_Test_Nonlinear 2D-Tracking/BO_Tracking_Test_Nonlinear.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(BO_Tracking_Test_Nonlinear
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
target_include_directories(BO_Tracking_Test_Nonlinear
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for BayesOpt tracking with CNEES
add_executable(BO_Tracking_Test_linear 2D-Tracking/BO_Tracking_Test_linear.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(BO_Tracking_Test_linear
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
target_include_directories(BO_Tracking_Test_linear
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for validating filter performance
add_executable(Validate_QR 2D-Tracking/Validate_QR.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(Validate_QR
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

target_include_directories(Validate_QR
    PRIVATE
    ${BAYESOPT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for demonstrating tuned parameters
add_executable(demo_linear_tracking 2D-Tracking/demo_linear_tracking.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(demo_linear_tracking
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(demo_linear_tracking
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for pure measurement nonlinear tracking demo (no true state initialization)
add_executable(demo_nonlinear_pure_measurement 2D-Tracking/demo_nonlinear_pure_measurement.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(demo_nonlinear_pure_measurement
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(demo_nonlinear_pure_measurement
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for testing system matrices and Proposition 3
add_executable(test_system_matrices 2D-Tracking/Testing/test_system_matrices.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(test_system_matrices
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
)

target_include_directories(test_system_matrices
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for zero noise validation using existing pipeline (Option B)
add_executable(test_zero_noise_validation_v2 2D-Tracking/Testing/test_zero_noise_validation_v2.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(test_zero_noise_validation_v2
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
)

target_include_directories(test_zero_noise_validation_v2
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)


# Executable for testing tuned parameters performance
add_executable(test_tuned_parameters_performance 2D-Tracking/Testing/test_tuned_parameters_performance.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(test_tuned_parameters_performance
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(test_tuned_parameters_performance
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for extreme parameter validation test
add_executable(test_extreme_parameter_validation 2D-Tracking/Testing/test_extreme_parameter_validation.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(test_extreme_parameter_validation
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(test_extreme_parameter_validation
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

# Executable for generating trajectory comparison data for visualization
add_executable(generate_trajectory_comparison_data 2D-Tracking/Testing/generate_trajectory_comparison_data.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(generate_trajectory_comparison_data
    PRIVATE
    Eigen3::Eigen
    g2o_solver_csparse
    g2o_csparse_extension
    g2o_core
    g2o_stuff
    g2o_solver_dense
    ${HDF5_LIBRARIES}
    yaml-cpp
)

target_include_directories(generate_trajectory_comparison_data
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)


# Flexible-Tracking folder ####################################################
# NOTE: Flexible-Tracking directory does not exist - commenting out build targets

# # Executable for generating flexible tracking data (all motion models)
# add_executable(flexible_data_generator Flexible-Tracking/flexible_data_generator.cpp)
# target_link_libraries(flexible_data_generator
#     PRIVATE
#     Eigen3::Eigen
#     ${HDF5_LIBRARIES}
#     yaml-cpp
# )
# target_include_directories(flexible_data_generator
#     PRIVATE
#     ${EIGEN3_INCLUDE_DIR}
#     ${HDF5_INCLUDE_DIRS}
# )

# # Executable for flexible Bayesian Optimization tracking
# add_executable(flexible_bo_tracking Flexible-Tracking/flexible_bo_tracking.cpp Flexible-Tracking/fg_flexible_tracking.cpp)
# target_link_libraries(flexible_bo_tracking
#     PRIVATE
#     ${BAYESOPT_LIBRARIES}
#     Eigen3::Eigen
#     OpenMP::OpenMP_CXX
#     g2o_solver_csparse
#     g2o_csparse_extension
#     g2o_core
#     g2o_stuff
#     g2o_solver_dense
#     ${HDF5_LIBRARIES}
#     yaml-cpp
# )
# target_include_directories(flexible_bo_tracking
#     PRIVATE
#     ${BAYESOPT_INCLUDE_DIRS}
#     ${EIGEN3_INCLUDE_DIR}
#     ${G2O_INCLUDE_DIRS}
#     /usr/local/include
#     ${HDF5_INCLUDE_DIRS}
# )

# # Executable for flexible tracking demo with optimal parameters
# add_executable(flexible_demo Flexible-Tracking/flexible_demo.cpp Flexible-Tracking/fg_flexible_tracking.cpp)
# target_link_libraries(flexible_demo
#     PRIVATE
#     Eigen3::Eigen
#     g2o_solver_csparse
#     g2o_csparse_extension
#     g2o_core
#     g2o_stuff
#     g2o_solver_dense
#     ${HDF5_LIBRARIES}
#     yaml-cpp
# )
# target_include_directories(flexible_demo
#     PRIVATE
#     ${EIGEN3_INCLUDE_DIR}
#     ${G2O_INCLUDE_DIRS}
#     /usr/local/include
#     ${HDF5_INCLUDE_DIRS}
# )

# # Executable for testing flexible factor graph
# add_executable(test_flexible_fg Flexible-Tracking/test_flexible_fg.cpp Flexible-Tracking/fg_flexible_tracking.cpp)
# target_link_libraries(test_flexible_fg
#     PRIVATE
#     Eigen3::Eigen
#     g2o_solver_csparse
#     g2o_csparse_extension
#     g2o_core
#     g2o_stuff
#     g2o_solver_dense
#     ${HDF5_LIBRARIES}
# )
# target_include_directories(test_flexible_fg
#     PRIVATE
#     ${EIGEN3_INCLUDE_DIR}
#     ${G2O_INCLUDE_DIRS}
#     /usr/local/include
#     ${HDF5_INCLUDE_DIRS}
# )
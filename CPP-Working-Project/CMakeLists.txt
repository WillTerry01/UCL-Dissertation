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
add_executable(tracking_gen_data 2D-Tracking/tracking_gen_data.cpp)
target_link_libraries(tracking_gen_data
    PRIVATE
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    yaml-cpp
)
target_include_directories(tracking_gen_data
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Executable for BayesOpt tracking with CNEES
add_executable(BO_Tracking_CNEES 2D-Tracking/BO_Tracking_CNEES.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(BO_Tracking_CNEES
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
target_include_directories(BO_Tracking_CNEES
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
add_executable(demo_tuned_parameters 2D-Tracking/demo_tuned_parameters.cpp 2D-Tracking/fg_class_tracking.cpp)
target_link_libraries(demo_tuned_parameters
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

target_include_directories(demo_tuned_parameters
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${G2O_INCLUDE_DIRS}
    /usr/local/include
    ${HDF5_INCLUDE_DIRS}
)

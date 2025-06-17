cmake_minimum_required(VERSION 3.10)
project(g2o_1d_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)

# Add executables
add_executable(1d_example 1D_example.cpp)
add_executable(2d_example 2D_example.cpp)

# Link libraries and include directories for both executables
foreach(target 1d_example 2d_example)
    target_include_directories(${target} PRIVATE 
        ${EIGEN3_INCLUDE_DIR}
        ${G2O_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE
        g2o_core
        g2o_stuff
        Eigen3::Eigen
    )
endforeach()

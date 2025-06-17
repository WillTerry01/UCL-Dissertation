cmake_minimum_required(VERSION 3.10)
project(tutorial_slam2d)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)

add_library(tutorial_slam2d_library ${G2O_LIB_TYPE}
  edge_se2_pointxy.cpp  se2.h                vertex_point_xy.h
  edge_se2.cpp    edge_se2_pointxy.h    vertex_se2.cpp
  edge_se2.h      vertex_point_xy.cpp  vertex_se2.h
  parameter_se2_offset.h parameter_se2_offset.cpp
  types_tutorial_slam2d.h types_tutorial_slam2d.cpp
  simulator.h simulator.cpp
  g2o_tutorial_slam2d_api.h
)

set_target_properties(tutorial_slam2d_library PROPERTIES OUTPUT_NAME ${LIB_PREFIX}tutorial_slam2d)
set_target_properties(tutorial_slam2d_library PROPERTIES
  VERSION ${G2O_LIB_VERSION}
  SOVERSION ${G2O_LIB_SOVERSION})

target_include_directories(tutorial_slam2d_library PRIVATE 
  ${EIGEN3_INCLUDE_DIR}
  ${G2O_INCLUDE_DIRS}
)
target_link_libraries(tutorial_slam2d_library 
  g2o_core
  g2o_stuff
  Eigen3::Eigen
)

add_executable(tutorial_slam2d
  tutorial_slam2d.cpp
)

set_target_properties(tutorial_slam2d PROPERTIES OUTPUT_NAME tutorial_slam2d)

target_include_directories(tutorial_slam2d PRIVATE 
  ${EIGEN3_INCLUDE_DIR}
  ${G2O_INCLUDE_DIRS}
)
target_link_libraries(tutorial_slam2d 
  tutorial_slam2d_library 
  g2o_core
  g2o_stuff
  Eigen3::Eigen
)


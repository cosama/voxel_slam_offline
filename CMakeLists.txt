cmake_minimum_required(VERSION 3.22)
project(voxel_slam_offline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree filters registration)
find_library(GTSAM_LIBRARY NAMES gtsam REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings.cpp)

target_compile_definitions(_core PRIVATE VOXELSLAM_PYTHON_MODULE)
target_compile_options(_core PRIVATE -O3 -pthread)
target_include_directories(_core PRIVATE
  cpp/include
  cpp/ros_compat/include
  cpp/upstream_voxelslam
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  ${GTSAM_LIBRARY}
)

install(TARGETS _core LIBRARY DESTINATION voxelslam)

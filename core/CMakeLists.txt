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

# Patch pinned upstream copy.
get_filename_component(VOXELSLAM_BRIDGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(UPSTREAM_VOXELSLAM "${VOXELSLAM_BRIDGE_ROOT}/upstream/Voxel-SLAM/VoxelSLAM")
if(NOT EXISTS "${UPSTREAM_VOXELSLAM}/src/voxelslam.cpp")
  message(FATAL_ERROR
    "Pinned upstream Voxel-SLAM source is missing at ${UPSTREAM_VOXELSLAM}. "
    "Run: git submodule update --init upstream/Voxel-SLAM")
endif()

set(VOXELSLAM_STAGE "${CMAKE_CURRENT_BINARY_DIR}/upstream_staged")
file(REMOVE_RECURSE "${VOXELSLAM_STAGE}")
file(MAKE_DIRECTORY "${VOXELSLAM_STAGE}")
file(COPY "${UPSTREAM_VOXELSLAM}/src" DESTINATION "${VOXELSLAM_STAGE}")

find_program(PATCH_EXECUTABLE patch REQUIRED)
file(GLOB VOXELSLAM_INTEGRATION_PATCHES
     "${CMAKE_CURRENT_SOURCE_DIR}/patches/integration/*.patch")
list(SORT VOXELSLAM_INTEGRATION_PATCHES)
list(LENGTH VOXELSLAM_INTEGRATION_PATCHES VOXELSLAM_PATCH_COUNT)
if(VOXELSLAM_PATCH_COUNT EQUAL 0)
  message(FATAL_ERROR "No patches found under patches/integration; the delta must be explicit")
endif()

foreach(patch_file IN LISTS VOXELSLAM_INTEGRATION_PATCHES)
  execute_process(
    COMMAND "${PATCH_EXECUTABLE}" -F0 --silent --no-backup-if-mismatch -p0 -i "${patch_file}"
    WORKING_DIRECTORY "${VOXELSLAM_STAGE}"
    RESULT_VARIABLE patch_result
    ERROR_VARIABLE patch_error)
  if(NOT patch_result EQUAL 0)
    message(FATAL_ERROR
      "Patch did not apply to pinned upstream: ${patch_file}\n${patch_error}\n"
      "The upstream revision has drifted from the pin recorded in UPSTREAM.md.")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch_file}")
endforeach()
message(STATUS "Voxel-SLAM: applied ${VOXELSLAM_PATCH_COUNT} patches to staged upstream")

pybind11_add_module(_core cpp/bindings.cpp)

target_compile_definitions(_core PRIVATE VOXELSLAM_PYTHON_MODULE)
target_compile_options(_core PRIVATE -O3 -pthread)
target_include_directories(_core PRIVATE
  cpp/include
  cpp/ros_compat/include
  "${VOXELSLAM_STAGE}/src"
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(_core PRIVATE
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  ${GTSAM_LIBRARY}
)

install(TARGETS _core LIBRARY DESTINATION voxelslam)

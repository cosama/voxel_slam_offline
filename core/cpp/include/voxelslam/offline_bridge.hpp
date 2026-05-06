#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace voxelslam_offline {

struct PoseRecord {
  double stamp = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
};

struct PointRecord {
  double stamp = 0.0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float intensity = 0.0f;
};

void reset_records(bool collect_map,
                   std::size_t max_map_points,
                   bool emit_deskewed_points,
                   std::size_t max_pending_deskewed_scans);
bool is_emitting_deskewed_points();
void record_pose(double stamp, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
void record_optimized_poses(const std::vector<PoseRecord>& poses);
void record_map_point(double stamp, float x, float y, float z, float intensity);
void record_dense_deskewed_points(const std::vector<PointRecord>& points);

}  // namespace voxelslam_offline

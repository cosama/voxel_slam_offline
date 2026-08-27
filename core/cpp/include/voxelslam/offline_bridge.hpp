#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
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

struct EventRecord {
  std::string type;
  double stamp = 0.0;
  std::uint64_t lidar_ticket = 0;
  std::string reason;
};

bool is_emitting_deskewed_points();
void record_lidar_processed(std::uint64_t ticket, bool pose_recorded);
void record_odometry_waiting_for_imu(std::uint64_t ticket, bool waiting);
void record_pose(double stamp, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
void record_optimized_poses(const std::vector<PoseRecord>& poses);
void record_dense_deskewed_points(const std::vector<PointRecord>& points);
void record_odometry_reset(double stamp, std::uint64_t lidar_ticket, const std::string& reason);

// Externally supplied odometry prior (a trajectory estimated by another
// frontend) that stands in for the IEKF pose of each sweep. No prior is
// installed unless the host provides one, and a stamp outside the prior's
// time span reports failure so the caller can fall back to upstream.
void record_prior_applied(bool degenerate_fallback);
void set_prior_trajectory(std::vector<PoseRecord> poses);
bool has_prior_trajectory();
bool prior_covers(double stamp_begin, double stamp_end);
bool prior_pose_at(double stamp,
                   Eigen::Vector3d* position,
                   Eigen::Quaterniond* orientation,
                   Eigen::Vector3d* velocity);
void record_loop_candidate(double score);
void record_loop_score_passed();
void record_loop_icp_result(double eig0,
                            double eig1,
                            double eig2,
                            int converged,
                            bool passed,
                            int match_count);
void record_loop_drift_ratio(double ratio, bool passed);
void record_loop_edge_added();
void record_loop_graph_optimization(std::size_t pose_count);
void record_loop_update_applied();
void record_gba_started(int keyframes);
void record_gba_completed(double runtime_seconds,
                          std::size_t pose_count,
                          std::size_t stage1_edges,
                          std::size_t stage2_edges);

template <typename ScanPoseBuffersT, typename IdsT>
void record_optimized_path(ScanPoseBuffersT& scan_pose_buffers, IdsT& ids) {
  std::vector<PoseRecord> poses;
  for (int map_index : ids) {
    for (auto* scan_pose : *scan_pose_buffers[map_index]) {
      Eigen::Quaterniond q(scan_pose->x.R);
      poses.push_back({
          scan_pose->x.t,
          scan_pose->x.p.x(),
          scan_pose->x.p.y(),
          scan_pose->x.p.z(),
          q.x(),
          q.y(),
          q.z(),
          q.w(),
      });
    }
  }
  record_optimized_poses(poses);
}

// Advance `x` from the previously accepted state `x_prev` by the prior's motion
// between the two sweeps. Deliberately relative: the prior's world frame is
// anchored by its own gravity/yaw initialization and does not coincide with
// upstream's, so imposing absolute poses would jump the estimate into a foreign
// frame. Only the body frame has to agree, and both are the IMU link.
template <typename StateT>
bool apply_prior_delta(StateT& x, const StateT& x_prev, double stamp_prev, double stamp) {
  Eigen::Vector3d p_prev;
  Eigen::Vector3d p_now;
  Eigen::Vector3d v_now;
  Eigen::Quaterniond q_prev;
  Eigen::Quaterniond q_now;
  if (!prior_pose_at(stamp_prev, &p_prev, &q_prev, nullptr) ||
      !prior_pose_at(stamp, &p_now, &q_now, &v_now)) {
    return false;
  }

  const Eigen::Quaterniond q_prev_inv = q_prev.conjugate();
  const Eigen::Quaterniond delta_q = q_prev_inv * q_now;
  const Eigen::Vector3d delta_p = q_prev_inv * (p_now - p_prev);

  const Eigen::Matrix3d R_prev = x_prev.R;
  x.R = R_prev * delta_q.toRotationMatrix();
  x.p = x_prev.p + R_prev * delta_p;
  // The prior reports velocity in its own world frame; rotate it through the
  // same alignment that carries the prior's frame onto upstream's at t_prev.
  x.v = R_prev * (q_prev_inv * v_now);
  return true;
}

template <typename CloudT, typename StateT, typename ExtrinsicT>
void record_dense_deskewed_scan(CloudT& scan, StateT& x_scan_end, ExtrinsicT& extrin_para) {
  (void)extrin_para;
  if (!is_emitting_deskewed_points() || scan.empty()) {
    return;
  }

  double max_offset = 0.0;
  for (auto& point : scan.points) {
    if (std::isfinite(point.curvature)) {
      max_offset = std::max(max_offset, static_cast<double>(point.curvature));
    }
  }

  const double scan_start_time = x_scan_end.t - max_offset;
  std::vector<PointRecord> points;
  points.reserve(scan.size());
  for (auto& point : scan.points) {
    points.push_back({
        scan_start_time + static_cast<double>(point.curvature),
        point.x,
        point.y,
        point.z,
        point.intensity,
    });
  }
  record_dense_deskewed_points(points);
}

}  // namespace voxelslam_offline

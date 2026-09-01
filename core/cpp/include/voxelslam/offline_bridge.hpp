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

struct PriorSweepRecord {
  double stamp = 0.0;
  double eigenvalue_min = 0.0;
  double eigenvalue_mid = 0.0;
  double eigenvalue_max = 0.0;
  int match_count = 0;
  bool prior_primed = false;
  bool degenerate = false;
};

bool is_emitting_deskewed_points();
void record_lidar_processed(std::uint64_t ticket, bool pose_recorded);
void record_odometry_waiting_for_imu(std::uint64_t ticket, bool waiting, double scan_end_time);
void record_odometry_waiting_for_prior(std::uint64_t ticket,
                                       bool waiting,
                                       double scan_begin_time,
                                       double scan_end_time);
void record_pose(double stamp, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
void record_optimized_poses(const std::vector<PoseRecord>& poses);
void record_dense_deskewed_points(const std::vector<PointRecord>& points);
void record_odometry_reset(double stamp, std::uint64_t lidar_ticket, const std::string& reason);

void record_prior_applied(bool degenerate_fallback);
void reset_prior(bool enabled);
void close_prior();
std::uint64_t push_prior_pose(double stamp,
                              const Eigen::Vector3d& position,
                              const Eigen::Quaterniond& orientation,
                              const Eigen::Matrix<double, 6, 6>* covariance);
void retain_prior_from(double stamp);
bool prior_enabled();
std::size_t prior_buffer_size();
bool prior_ready_for_sweep(double stamp_begin, double stamp_end);
bool prior_covers(double stamp_begin, double stamp_end);
bool prior_pose_at(double stamp,
                   Eigen::Vector3d* position,
                   Eigen::Quaterniond* orientation,
                   Eigen::Vector3d* velocity);
bool prior_pose_covariance(Eigen::Matrix<double, 6, 6>* covariance);

bool prior_relative_motion(double stamp_begin,
                           double stamp_end,
                           Eigen::Matrix3d* rotation,
                           Eigen::Vector3d* translation);

void set_prior_ba_sigma(double sigma_rot_rad, double sigma_pos_m);
bool prior_ba_sigma(double* sigma_rot_rad, double* sigma_pos_m);

void set_prior_deskew(bool enabled);
bool prior_deskew_enabled();

struct PriorDeskewFrame {
  std::vector<double> stamps;
  std::vector<Eigen::Vector3d> positions;
  std::vector<Eigen::Quaterniond> orientations;
  Eigen::Quaterniond reference_rotation_inverse = Eigen::Quaterniond::Identity();
  Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
};
bool prior_deskew_frame(double stamp_begin,
                        double stamp_reference,
                        PriorDeskewFrame* frame);

void prior_deskew_point(const PriorDeskewFrame& frame,
                        double stamp,
                        Eigen::Matrix3d* rotation,
                        Eigen::Vector3d* translation);

void record_prior_sweep_values(double stamp,
                               const Eigen::Vector3d& nnt_eigenvalues,
                               int match_count,
                               bool prior_primed,
                               bool degenerate);
std::vector<PriorSweepRecord> prior_sweep_trace();

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
  x.v = R_prev * (q_prev_inv * v_now);

  Eigen::Matrix<double, 6, 6> prior_cov;
  if (prior_pose_covariance(&prior_cov)) {
    const Eigen::Index n = x.cov.rows();
    x.cov.block(0, 0, 6, n).setZero();
    x.cov.block(0, 0, n, 6).setZero();
    Eigen::Matrix<double, 6, 6> frame = Eigen::Matrix<double, 6, 6>::Identity();
    frame.block<3, 3>(3, 3) = R_prev * q_prev_inv.toRotationMatrix();
    x.cov.block(0, 0, 6, 6) = frame * prior_cov * frame.transpose();
  }
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

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

// One row per steady-state sweep of what the odometry IEKF already computes and
// upstream discards. Two things: how well the matched plane normals spanned the
// space, which is what upstream's degeneracy verdict is a threshold on, and how
// far point-to-plane registration moved the odometry prior's prediction, which
// is the only direct measure of what the backend adds over the prior.
//
// Diagnostic. Nothing in the estimator reads these back; see RVS_STATUS.md
// Part II for why an earlier attempt to close that loop was removed.
struct PriorSweepRecord {
  double stamp = 0.0;
  // Eigenvalues of `nnt`, the outer-product sum of the matched plane normals,
  // ascending, and how many points matched. Upstream collapses the first to
  // `eigenvalues[0] < 14` and discards both.
  //
  // Read the triple scale-free, as eigen_min/eigen_max. `nnt` is a sum over
  // matched points, so its raw scale says how many points matched as much as it
  // says how well their normals span the space: across this suite the median
  // `eigenvalues[0]` ranges 65 to 859 while the ratio ranges 0.075 to 0.54.
  double eigenvalue_min = 0.0;
  double eigenvalue_mid = 0.0;
  double eigenvalue_max = 0.0;
  int match_count = 0;
  // Whether the prior predicted this sweep, and whether the IEKF's verdict on
  // its own update was degenerate.
  bool prior_primed = false;
  bool degenerate = false;
  // How far registration moved the prior's prediction, as magnitudes in the
  // predicted body frame: radians and metres. Taken before the degenerate
  // fallback overwrites the state with the prediction itself, which would
  // otherwise read as perfect agreement on exactly the sweeps where the LiDAR
  // had nothing to say -- so this is not recoverable after the fact by
  // differencing the two trajectories.
  double disagreement_rot = 0.0;
  double disagreement_pos = 0.0;
  // What `apply_prior_delta` overwrites: the IMU-propagated prediction
  // covariance as it stood before the prior's pose block replaced it. See
  // PropagatedCovariance for what the three numbers are for.
  double propagated_sigma_rot = 0.0;
  double propagated_sigma_pos = 0.0;
  double propagated_pose_velocity_correlation = 0.0;
  double propagated_pose_bias_correlation = 0.0;
};

// A summary of the IMU-propagated prediction covariance, taken per sweep before
// `apply_prior_delta` replaces its pose block. Diagnostic; nothing reads it
// back. It exists to answer two questions cheaply, both of which have to be
// answered before anyone reworks how the prior is weighted:
//
// 1. Does the propagated covariance move? The prior's weight is a hand-tuned
//    constant standing in for it. If a steady-state IEKF settles and sits
//    there, a constant was never wrong and the question closes. If it moves,
//    the prior's fixed sigma means something different on every sweep.
// 2. How much does zeroing the cross-covariance discard? Replacing the pose
//    block forces the rest to be cleared -- shrinking a diagonal block while
//    keeping its off-diagonals generally destroys positive-definiteness, and
//    `cov.inverse()` would then be silently meaningless. Reported as two
//    correlations, because they answer different questions and only one of them
//    is a loss:
//    - against velocity, the coupling is kinematic (over one sweep dp ~ dv*dt),
//      and `apply_prior_delta` replaces the velocity from the prior too, so
//      clearing a correlation between two quantities that were both replaced is
//      consistent rather than lossy.
//    - against the biases is the channel through which a pose measurement
//      corrects IMU bias in a filter. Clearing that is the real cost. The local
//      BA still estimates the biases jointly over its window, so what is lost is
//      a per-sweep refinement, not observability.
struct PropagatedCovariance {
  // RMS one-sigma over the three axes of each pose block.
  double sigma_rot_rad = 0.0;
  double sigma_pos_m = 0.0;
  // Largest absolute correlation coefficient between a pose DOF and, in turn,
  // the velocity and the two bias blocks. Scale-free, so comparable across
  // sweeps and runs.
  double max_pose_velocity_correlation = 0.0;
  double max_pose_bias_correlation = 0.0;
};

template <typename MatrixT>
PropagatedCovariance summarize_propagated_covariance(const MatrixT& cov) {
  PropagatedCovariance out;
  const Eigen::Index n = cov.rows();
  if (n < 6) {
    return out;
  }
  double rot = 0.0;
  double pos = 0.0;
  for (Eigen::Index i = 0; i < 3; ++i) {
    rot += std::max(0.0, cov(i, i));
    pos += std::max(0.0, cov(i + 3, i + 3));
  }
  out.sigma_rot_rad = std::sqrt(rot / 3.0);
  out.sigma_pos_m = std::sqrt(pos / 3.0);
  for (Eigen::Index i = 0; i < 6; ++i) {
    for (Eigen::Index j = 6; j < n; ++j) {
      const double denom = cov(i, i) * cov(j, j);
      if (!(denom > 0.0)) {
        continue;
      }
      const double correlation = std::abs(cov(i, j)) / std::sqrt(denom);
      double& target = j < 9 ? out.max_pose_velocity_correlation
                             : out.max_pose_bias_correlation;
      target = std::max(target, correlation);
    }
  }
  return out;
}

bool is_emitting_deskewed_points();
void record_lidar_processed(std::uint64_t ticket, bool pose_recorded);
// Report whether the estimator has parked `ticket` because the IMU stream does
// not yet reach past `scan_end_time`. The end time is part of the report so the
// host can re-evaluate the same predicate against the data it has submitted,
// rather than trusting when the worker last looked.
void record_odometry_waiting_for_imu(std::uint64_t ticket, bool waiting, double scan_end_time);
// The same for the odometry prior. Both sweep bounds are reported, not just
// the end, because the readiness predicate is a function of both: a sweep that
// begins before the prior's first sample can never be repaired by a future
// push and is released immediately, while a missing right bracket must park.
// Publishing the pair lets the host re-evaluate `prior_ready_for_sweep()`
// itself rather than maintain a second, subtly different rule.
void record_odometry_waiting_for_prior(std::uint64_t ticket,
                                       bool waiting,
                                       double scan_begin_time,
                                       double scan_end_time);
void record_pose(double stamp, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
void record_optimized_poses(const std::vector<PoseRecord>& poses);
void record_dense_deskewed_points(const std::vector<PointRecord>& points);
void record_odometry_reset(double stamp, std::uint64_t lidar_ticket, const std::string& reason);

// Externally supplied odometry prior (a trajectory estimated by another
// frontend) that stands in for the IEKF pose of each sweep. Samples arrive
// causally beside IMU/LiDAR input. Queries never extrapolate: the odometry
// worker parks until the right interpolation bracket's central-difference
// velocity is final (one additional sample beyond that bracket).
void record_prior_applied(bool degenerate_fallback);
void reset_prior(bool enabled);
// Declare the prior stream complete. Every sweep past its last sample is then
// released to upstream prediction immediately instead of parking for lookahead
// that will never arrive, and the last sample's velocity becomes final as a
// one-sided difference. A producer that knows it has no more samples must call
// this: without it the host's per-sweep barrier silently stops applying for the
// remainder of the run, and the sweeps queue until finish() drains them.
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
// How accurate the prior's between-sweep motion is, as one-sigma. Both
// estimators regularize toward the predicted state with weight `cov.inverse()`,
// so this is where the prior stops being a guess and becomes a measurement.
// Reports false when no sigma is configured, which leaves the IMU-propagated
// covariance in place and reproduces the earlier behavior exactly.
//
// Fixed for the run, deliberately. Making it move per sweep with the LiDAR's
// conditioning was built and measured and does nothing, because it is already
// in the equation: `jac.tail(3)` is the plane normal, so `H_T_H`'s translation
// block is `nnt` weighted per point, and `K_1` therefore already hands the
// prior the directions the LiDAR cannot see. The two sides are not even
// independent -- this covariance feeds `var_world`, which gates plane matching
// and sets `R_inv`, so moving one side of the ratio moves the other. See
// RVS_STATUS.md Part II.
bool prior_pose_covariance(Eigen::Matrix<double, 6, 6>* covariance);

// The prior's motion between two sweep stamps, expressed in the body frame of
// the earlier one -- the same quantity the local BA's window states carry
// between consecutive poses, so the arbitrary world-frame offset between the
// prior and upstream cancels and only the shared IMU body frame is assumed.
bool prior_relative_motion(double stamp_begin,
                           double stamp_end,
                           Eigen::Matrix3d* rotation,
                           Eigen::Vector3d* translation);

// How accurate that between-sweep motion is, as the one-sigma weight the local
// BA gives it against the LiDAR plane factors and the IMU preintegration.
// Reports false when unset, which leaves the local BA exactly as upstream wrote
// it. Fixed for the run: the adaptive weight below does not reach the local BA,
// for the reason given there.
void set_prior_ba_sigma(double sigma_rot_rad, double sigma_pos_m);
bool prior_ba_sigma(double* sigma_rot_rad, double* sigma_pos_m);

// Undistort each point against the prior's own trajectory instead of IMU dead
// reckoning. Deskew happens before a scan is ever voxelized, so its error is
// baked into the map and no amount of later pose refinement removes it -- which
// makes it the one sharpening lever that costs nothing on the trajectory. A
// run-long choice of source: the adaptive weight below does not reach it, for
// the reason given there.
void set_prior_deskew(bool enabled);
bool prior_deskew_enabled();

// One sweep's worth of the prior, copied out under the prior lock so the
// per-point path takes no lock and searches only a handful of samples.
//
// `prior_deskew_frame()` reports false unless the prior spans the whole sweep,
// so the caller either undistorts every point against the prior or none of
// them. Mixing the two within one scan is worse than not using the prior at
// all: a point left uncorrected is not half-corrected, it is asserted to be at
// the sweep-end pose already, and that error is baked into the voxel map.
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

// The prior's motion from `stamp` to the frame's reference stamp, in the
// reference's body frame -- the same pair upstream builds from `imu_poses`, so
// the prior's world frame cancels here as it does everywhere else. `stamp` is
// clamped to the frame's span, which the frame guarantees covers the sweep.
void prior_deskew_point(const PriorDeskewFrame& frame,
                        double stamp,
                        Eigen::Matrix3d* rotation,
                        Eigen::Vector3d* translation);

// Append one sweep's trace row. Called from the odometry thread, once per
// steady-state sweep, after registration.
void record_prior_sweep_values(double stamp,
                               const Eigen::Vector3d& nnt_eigenvalues,
                               int match_count,
                               bool prior_primed,
                               bool degenerate,
                               double disagreement_rot,
                               double disagreement_pos,
                               const PropagatedCovariance& propagated);
std::vector<PriorSweepRecord> prior_sweep_trace();

// The same, taking the two states directly: the pose the prior predicted and
// the pose registration returned. The disagreement is their relative transform
// expressed in the predicted body frame, so it is a per-sweep local quantity
// that does not accumulate as the two trajectories separate globally, and the
// body frame is the one thing the prior and upstream are known to share.
template <typename StateT>
void record_prior_sweep(const StateT& x_predicted,
                        const StateT& x_registered,
                        const Eigen::Vector3d& nnt_eigenvalues,
                        int match_count,
                        bool prior_primed,
                        bool degenerate,
                        const PropagatedCovariance& propagated) {
  const Eigen::AngleAxisd delta(
      Eigen::Matrix3d(x_predicted.R.transpose() * x_registered.R));
  const Eigen::Vector3d offset =
      x_predicted.R.transpose() * (x_registered.p - x_predicted.p);
  record_prior_sweep_values(x_registered.t, nnt_eigenvalues, match_count,
                            prior_primed, degenerate, std::abs(delta.angle()),
                            offset.norm(), propagated);
}

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

  // Tell the estimator how good that prediction is. Both `lio_state_estimation`
  // and `lio_state_estimation_kdtree` form `cov_inv = x_curr.cov.inverse()` and
  // fold it into `K_1 = (H_T_H + cov_inv).inverse()`, so the prediction's
  // covariance is exactly the weight the prior carries against the
  // point-to-plane hessian. Leaving the IMU-propagated covariance there claims
  // the warm start is as uncertain as dead reckoning, and LiDAR then wins
  // outright -- including in the directions where `H_T_H` is rank deficient and
  // LiDAR has nothing to say. Substituting the prior's own between-sweep
  // accuracy makes the handoff continuous and directional instead of hanging on
  // the binary `min_eigen_value` verdict: LiDAR still moves the pose wherever it
  // is informative, and the prior holds the rest.
  //
  // The pose block is replaced rather than merged, and its cross-covariance with
  // velocity and the biases is cleared, so the result stays block diagonal and
  // positive definite. Rotation is a body-frame right perturbation and position
  // is a world-frame offset, matching `IMUST::operator+=`, so isotropic blocks
  // are the right shape for both.
  Eigen::Matrix<double, 6, 6> prior_cov;
  if (prior_pose_covariance(&prior_cov)) {
    const Eigen::Index n = x.cov.rows();
    x.cov.block(0, 0, 6, n).setZero();
    x.cov.block(0, 0, n, 6).setZero();
    // Rotation uncertainty is a body-frame right perturbation in both
    // frontends. Position uncertainty is expressed in the prior world frame;
    // rotate that half through the same frame alignment used for the delta.
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

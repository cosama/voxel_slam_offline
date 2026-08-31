#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <Eigen/Cholesky>

#include <voxelslam/offline_bridge.hpp>

#include "BTC.cpp"
#include "voxelslam.cpp"

namespace py = pybind11;

namespace voxelslam_offline {

struct ScalarStats {
  std::uint64_t count = 0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();
  double sum = 0.0;

  void observe(double value) {
    if (!std::isfinite(value)) {
      return;
    }
    ++count;
    min = std::min(min, value);
    max = std::max(max, value);
    sum += value;
  }
};

struct Metrics {
  std::uint64_t odometry_degrade_resets = 0;
  // How much of the run the odometry prior actually decided. `applied` counts
  // sweeps it predicted; `fallback` counts the subset where the IEKF reported a
  // degenerate update and the prior's pose stood unregistered. A high fallback
  // share is the signature of a run the prior carried rather than one the map
  // constrained.
  std::uint64_t odometry_prior_applied = 0;
  std::uint64_t odometry_prior_fallback = 0;
  // The odometry IEKF's own conditioning and its disagreement with the prior,
  // aggregated. The full per-sweep rows are in the run's prior_trace; these are
  // here so a manifest carries the shape of the run without it.
  ScalarStats iekf_eigenvalue_min;
  ScalarStats iekf_eigenvalue_ratio;
  ScalarStats iekf_match_count;
  ScalarStats prior_disagreement_rot;
  ScalarStats prior_disagreement_pos;
  // The IMU-propagated prediction covariance the prior's pose block replaces,
  // and how strongly that pose block was correlated with velocity and the
  // biases before the substitution cleared those cross terms. See
  // voxelslam_offline::PropagatedCovariance for why both are worth knowing.
  ScalarStats propagated_sigma_rot;
  ScalarStats propagated_sigma_pos;
  ScalarStats propagated_pose_velocity_correlation;
  ScalarStats propagated_pose_bias_correlation;

  std::uint64_t loop_candidates = 0;
  std::uint64_t loop_score_passed = 0;
  std::uint64_t loop_icp_converged = 0;
  std::uint64_t loop_icp_passed = 0;
  std::uint64_t loop_icp_failed = 0;
  std::uint64_t loop_drift_passed = 0;
  std::uint64_t loop_drift_rejected = 0;
  std::uint64_t loop_edges_added = 0;
  std::uint64_t loop_graph_optimizations = 0;
  std::uint64_t loop_updates_applied = 0;
  ScalarStats loop_score;
  ScalarStats loop_icp_match_count;
  ScalarStats loop_icp_eigen_min;
  ScalarStats loop_icp_eigen_mid;
  ScalarStats loop_icp_eigen_max;
  ScalarStats loop_drift_ratio;
  ScalarStats loop_graph_pose_count;

  std::uint64_t gba_started = 0;
  std::uint64_t gba_completed = 0;
  ScalarStats gba_keyframes;
  ScalarStats gba_runtime_seconds;
  ScalarStats gba_pose_count;
  ScalarStats gba_stage1_edges;
  ScalarStats gba_stage2_edges;
};

struct PipelineStatus {
  std::uint64_t latest_imu_ticket = 0;
  std::uint64_t latest_prior_ticket = 0;
  std::uint64_t latest_lidar_ticket = 0;
  std::uint64_t latest_lidar_processed_ticket = 0;
  std::uint64_t lidar_completed = 0;
  std::uint64_t lidar_completed_with_pose = 0;
  std::uint64_t lidar_completed_without_pose = 0;
  bool odometry_waiting_for_imu = false;
  std::uint64_t odometry_waiting_lidar_ticket = 0;
  // End time of the parked sweep, so the host can re-test the wait predicate
  // against the IMU it has actually submitted. See blocked_on_unsubmitted_imu().
  double odometry_waiting_scan_end_time = 0.0;
  bool odometry_waiting_for_prior = false;
  // Both bounds, so the host can re-run prior_ready_for_sweep() rather than
  // keep a second copy of the rule. See blocked_on_unsubmitted_prior().
  double odometry_waiting_prior_begin_time = 0.0;
  double odometry_waiting_prior_end_time = 0.0;
  bool odometry_thread_finished = false;
  bool loop_thread_finished = false;
  bool gba_thread_finished = false;
};

struct Recorder {
  std::mutex mutex;
  std::vector<PoseRecord> poses;
  std::vector<PoseRecord> optimized_poses;
  std::vector<EventRecord> events;
  std::vector<PriorSweepRecord> prior_sweeps;
  std::deque<std::vector<PointRecord>> deskewed_scans;
  Metrics metrics;
  PipelineStatus pipeline;
  bool emit_deskewed_points = false;
};

Recorder& recorder() {
  static Recorder instance;
  return instance;
}

void reset_records(bool emit_deskewed_points) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  std::vector<PoseRecord>().swap(rec.poses);
  std::vector<PoseRecord>().swap(rec.optimized_poses);
  std::vector<EventRecord>().swap(rec.events);
  std::vector<PriorSweepRecord>().swap(rec.prior_sweeps);
  rec.deskewed_scans.clear();
  rec.metrics = Metrics();
  rec.pipeline = PipelineStatus();
  rec.emit_deskewed_points = emit_deskewed_points;
}

bool is_emitting_deskewed_points() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.emit_deskewed_points;
}

void record_imu_pushed(std::uint64_t ticket) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.latest_imu_ticket = std::max(rec.pipeline.latest_imu_ticket, ticket);
}

void record_lidar_pushed(std::uint64_t ticket) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.latest_lidar_ticket = std::max(rec.pipeline.latest_lidar_ticket, ticket);
}

void record_lidar_processed(std::uint64_t ticket, bool pose_recorded) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.latest_lidar_processed_ticket = std::max(rec.pipeline.latest_lidar_processed_ticket, ticket);
  ++rec.pipeline.lidar_completed;
  if (pose_recorded) {
    ++rec.pipeline.lidar_completed_with_pose;
  } else {
    ++rec.pipeline.lidar_completed_without_pose;
  }
}

void record_odometry_waiting_for_imu(std::uint64_t ticket, bool waiting, double scan_end_time) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  // Published as one unit under the recorder mutex: a reader that sees the
  // waiting flag set always sees the end time that goes with it.
  rec.pipeline.odometry_waiting_for_imu = waiting;
  rec.pipeline.odometry_waiting_lidar_ticket = ticket;
  rec.pipeline.odometry_waiting_scan_end_time = scan_end_time;
}

void record_odometry_waiting_for_prior(std::uint64_t ticket,
                                       bool waiting,
                                       double scan_begin_time,
                                       double scan_end_time) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.odometry_waiting_for_prior = waiting;
  rec.pipeline.odometry_waiting_lidar_ticket = ticket;
  rec.pipeline.odometry_waiting_prior_begin_time = scan_begin_time;
  rec.pipeline.odometry_waiting_prior_end_time = scan_end_time;
}

void record_odometry_thread_finished() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.odometry_thread_finished = true;
}

void record_loop_thread_finished() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.loop_thread_finished = true;
}

void record_gba_thread_finished() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.gba_thread_finished = true;
}

void record_pose(double stamp, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.poses.push_back({
      stamp,
      position.x(),
      position.y(),
      position.z(),
      orientation.x(),
      orientation.y(),
      orientation.z(),
      orientation.w(),
  });
}

void record_optimized_poses(const std::vector<PoseRecord>& poses) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.optimized_poses = poses;
}

void record_dense_deskewed_points(const std::vector<PointRecord>& points) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  if (!rec.emit_deskewed_points || points.empty()) {
    return;
  }
  rec.deskewed_scans.push_back(points);
}

void record_odometry_reset(double stamp, std::uint64_t lidar_ticket, const std::string& reason) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.odometry_degrade_resets;
  rec.events.push_back({"odometry_reset", stamp, lidar_ticket, reason});
}

void record_loop_candidate(double score) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.loop_candidates;
  rec.metrics.loop_score.observe(score);
}

void record_loop_score_passed() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.loop_score_passed;
}

void record_loop_icp_result(double eig0,
                            double eig1,
                            double eig2,
                            int converged,
                            bool passed,
                            int match_count) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  auto& d = rec.metrics;
  if (converged) {
    ++d.loop_icp_converged;
  }
  if (passed) {
    ++d.loop_icp_passed;
  } else {
    ++d.loop_icp_failed;
  }
  d.loop_icp_match_count.observe(match_count);
  d.loop_icp_eigen_min.observe(eig0);
  d.loop_icp_eigen_mid.observe(eig1);
  d.loop_icp_eigen_max.observe(eig2);
}

void record_loop_drift_ratio(double ratio, bool passed) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  if (passed) {
    ++rec.metrics.loop_drift_passed;
  } else {
    ++rec.metrics.loop_drift_rejected;
  }
  rec.metrics.loop_drift_ratio.observe(ratio);
}

void record_loop_edge_added() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.loop_edges_added;
}

void record_loop_graph_optimization(std::size_t pose_count) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.loop_graph_optimizations;
  rec.metrics.loop_graph_pose_count.observe(static_cast<double>(pose_count));
}

void record_loop_update_applied() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.loop_updates_applied;
}

void record_gba_started(int keyframes) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.gba_started;
  rec.metrics.gba_keyframes.observe(keyframes);
}

void record_gba_completed(double runtime_seconds,
                          std::size_t pose_count,
                          std::size_t stage1_edges,
                          std::size_t stage2_edges) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  auto& d = rec.metrics;
  ++d.gba_completed;
  d.gba_runtime_seconds.observe(runtime_seconds);
  d.gba_pose_count.observe(static_cast<double>(pose_count));
  d.gba_stage1_edges.observe(static_cast<double>(stage1_edges));
  d.gba_stage2_edges.observe(static_cast<double>(stage2_edges));
}

std::vector<PoseRecord> best_poses(const Recorder& rec) {
  if (rec.optimized_poses.empty()) {
    return rec.poses;
  }

  std::vector<PoseRecord> poses = rec.optimized_poses;
  const double last_optimized_stamp = poses.back().stamp;
  for (const PoseRecord& pose : rec.poses) {
    if (pose.stamp > last_optimized_stamp) {
      poses.push_back(pose);
    }
  }
  return poses;
}

std::vector<PoseRecord> take_poses() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return best_poses(rec);
}

std::vector<PoseRecord> snapshot_best_poses() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return best_poses(rec);
}

std::vector<std::vector<PointRecord>> take_deskewed_scans() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  std::vector<std::vector<PointRecord>> scans;
  scans.reserve(rec.deskewed_scans.size());
  while (!rec.deskewed_scans.empty()) {
    scans.push_back(std::move(rec.deskewed_scans.front()));
    rec.deskewed_scans.pop_front();
  }
  return scans;
}

Metrics snapshot_metrics() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.metrics;
}

std::vector<EventRecord> snapshot_events() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.events;
}

PipelineStatus snapshot_pipeline_status() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.pipeline;
}


// ---------------------------------------------------------------------------
// Externally supplied odometry prior
// ---------------------------------------------------------------------------

struct PriorTrajectory {
  std::mutex mutex;
  std::vector<double> stamps;
  std::vector<Eigen::Vector3d> positions;
  std::vector<Eigen::Quaterniond> orientations;
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
  bool covariance_set = false;
  bool covariance_mode_known = false;
  bool enabled = false;
  bool closed = false;
  std::uint64_t latest_ticket = 0;
  double retain_from = -std::numeric_limits<double>::infinity();
};

PriorTrajectory& prior_trajectory() {
  static PriorTrajectory instance;
  return instance;
}

// The prior carries pose only, but upstream's state needs a velocity for IMU
// preintegration and deskew. Central difference at interior samples, one-sided
// at the buffer's two ends.
//
// Derived on demand rather than maintained in a parallel array: there is then
// no incremental update to keep in step with retention, and no sample whose
// value depends on how the producer batched its pushes. What makes a consumed
// velocity final is that the sample is interior, and that is exactly what
// `prior_ready_for_sweep()` waits for; after `close_prior()` the tail's
// one-sided difference is final too, because no further sample can arrive.
//
// Callers must hold `prior.mutex`.
Eigen::Vector3d prior_velocity_at(const PriorTrajectory& prior, std::size_t index) {
  const std::size_t count = prior.stamps.size();
  if (count < 2) {
    return Eigen::Vector3d::Zero();
  }
  const std::size_t lo = index == 0 ? 0 : index - 1;
  const std::size_t hi = index + 1 < count ? index + 1 : count - 1;
  const double dt = prior.stamps[hi] - prior.stamps[lo];
  if (!(dt > 0.0)) {
    return Eigen::Vector3d::Zero();
  }
  return (prior.positions[hi] - prior.positions[lo]) / dt;
}

void record_prior_applied(bool degenerate_fallback) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  ++rec.metrics.odometry_prior_applied;
  if (degenerate_fallback) {
    ++rec.metrics.odometry_prior_fallback;
  }
}

bool prior_pose_covariance(Eigen::Matrix<double, 6, 6>* covariance) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  if (!prior.covariance_set) {
    return false;
  }
  *covariance = prior.covariance;
  return true;
}

struct PriorBaSigma {
  std::atomic<double> rot{0.0};
  std::atomic<double> pos{0.0};
};

PriorBaSigma& prior_ba_sigma_state() {
  static PriorBaSigma instance;
  return instance;
}

void set_prior_ba_sigma(double sigma_rot_rad, double sigma_pos_m) {
  auto& s = prior_ba_sigma_state();
  s.rot.store(sigma_rot_rad > 0.0 ? sigma_rot_rad : 0.0);
  s.pos.store(sigma_pos_m > 0.0 ? sigma_pos_m : 0.0);
}

bool prior_ba_sigma(double* sigma_rot_rad, double* sigma_pos_m) {
  auto& s = prior_ba_sigma_state();
  const double rot = s.rot.load();
  const double pos = s.pos.load();
  if (!(rot > 0.0) || !(pos > 0.0)) {
    return false;
  }
  *sigma_rot_rad = rot;
  *sigma_pos_m = pos;
  return true;
}

std::atomic<bool>& prior_deskew_state() {
  static std::atomic<bool> instance{false};
  return instance;
}

void set_prior_deskew(bool enabled) { prior_deskew_state().store(enabled); }

bool prior_deskew_enabled() { return prior_deskew_state().load(); }

void record_prior_sweep_values(double stamp,
                               const Eigen::Vector3d& nnt_eigenvalues,
                               int match_count,
                               bool prior_primed,
                               bool degenerate,
                               double disagreement_rot,
                               double disagreement_pos,
                               const PropagatedCovariance& propagated) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  PriorSweepRecord row;
  row.stamp = stamp;
  row.eigenvalue_min = nnt_eigenvalues[0];
  row.eigenvalue_mid = nnt_eigenvalues[1];
  row.eigenvalue_max = nnt_eigenvalues[2];
  row.match_count = match_count;
  row.prior_primed = prior_primed;
  row.degenerate = degenerate;
  row.disagreement_rot = disagreement_rot;
  row.disagreement_pos = disagreement_pos;
  row.propagated_sigma_rot = propagated.sigma_rot_rad;
  row.propagated_sigma_pos = propagated.sigma_pos_m;
  row.propagated_pose_velocity_correlation = propagated.max_pose_velocity_correlation;
  row.propagated_pose_bias_correlation = propagated.max_pose_bias_correlation;
  rec.prior_sweeps.push_back(row);

  auto& d = rec.metrics;
  d.iekf_eigenvalue_min.observe(nnt_eigenvalues[0]);
  d.iekf_eigenvalue_ratio.observe(nnt_eigenvalues[2] > 0.0
                                      ? nnt_eigenvalues[0] / nnt_eigenvalues[2]
                                      : 0.0);
  d.iekf_match_count.observe(match_count);
  if (prior_primed) {
    d.prior_disagreement_rot.observe(disagreement_rot);
    d.prior_disagreement_pos.observe(disagreement_pos);
  }
  if (propagated.sigma_pos_m > 0.0) {
    d.propagated_sigma_rot.observe(propagated.sigma_rot_rad);
    d.propagated_sigma_pos.observe(propagated.sigma_pos_m);
    d.propagated_pose_velocity_correlation.observe(
        propagated.max_pose_velocity_correlation);
    d.propagated_pose_bias_correlation.observe(
        propagated.max_pose_bias_correlation);
  }
}

std::vector<PriorSweepRecord> prior_sweep_trace() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.prior_sweeps;
}

// Interpolate within a sweep's copied window. The same linear/slerp pair
// `prior_pose_at` uses, over the same bracket; `stamp` is clamped because the
// frame is built to span the sweep and a point outside it is a rounding edge,
// not a query the caller can do anything with.
void interpolate_deskew_frame(const PriorDeskewFrame& frame,
                              double stamp,
                              Eigen::Vector3d* position,
                              Eigen::Quaterniond* orientation) {
  const std::size_t count = frame.stamps.size();
  if (count == 1) {
    *position = frame.positions[0];
    *orientation = frame.orientations[0];
    return;
  }
  const auto upper = std::upper_bound(frame.stamps.begin(), frame.stamps.end(), stamp);
  std::size_t hi = static_cast<std::size_t>(upper - frame.stamps.begin());
  if (hi == 0) {
    hi = 1;
  }
  if (hi >= count) {
    hi = count - 1;
  }
  const std::size_t lo = hi - 1;
  const double span = frame.stamps[hi] - frame.stamps[lo];
  double u = span > 0.0 ? (stamp - frame.stamps[lo]) / span : 0.0;
  u = std::min(1.0, std::max(0.0, u));
  *position = frame.positions[lo] + u * (frame.positions[hi] - frame.positions[lo]);
  *orientation = frame.orientations[lo].slerp(u, frame.orientations[hi]);
}

bool prior_deskew_frame(double stamp_begin,
                        double stamp_reference,
                        PriorDeskewFrame* frame) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  const std::size_t count = prior.stamps.size();
  // All of the sweep or none of it: a point the prior cannot place must not be
  // left at its raw coordinates beside points that were undistorted, because
  // that asserts it was already at the sweep-end pose.
  if (count < 2 || !std::isfinite(stamp_begin) || !std::isfinite(stamp_reference) ||
      stamp_begin > stamp_reference || stamp_begin < prior.stamps.front() ||
      stamp_reference > prior.stamps.back()) {
    return false;
  }

  const std::size_t first =
      static_cast<std::size_t>(
          std::upper_bound(prior.stamps.begin(), prior.stamps.end(), stamp_begin) -
          prior.stamps.begin()) -
      1;
  const std::size_t last = static_cast<std::size_t>(
      std::lower_bound(prior.stamps.begin(), prior.stamps.end(), stamp_reference) -
      prior.stamps.begin());

  const std::size_t span = last - first + 1;
  frame->stamps.assign(prior.stamps.begin() + first, prior.stamps.begin() + first + span);
  frame->positions.assign(prior.positions.begin() + first,
                          prior.positions.begin() + first + span);
  frame->orientations.assign(prior.orientations.begin() + first,
                             prior.orientations.begin() + first + span);

  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  interpolate_deskew_frame(*frame, stamp_reference, &position, &orientation);
  frame->reference_rotation_inverse = orientation.conjugate();
  frame->reference_position = position;
  return true;
}

void prior_deskew_point(const PriorDeskewFrame& frame,
                        double stamp,
                        Eigen::Matrix3d* rotation,
                        Eigen::Vector3d* translation) {
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  interpolate_deskew_frame(frame, stamp, &position, &orientation);
  *rotation = (frame.reference_rotation_inverse * orientation).toRotationMatrix();
  *translation =
      frame.reference_rotation_inverse * (position - frame.reference_position);
}

bool prior_relative_motion(double stamp_begin,
                           double stamp_end,
                           Eigen::Matrix3d* rotation,
                           Eigen::Vector3d* translation) {
  Eigen::Vector3d p_begin;
  Eigen::Vector3d p_end;
  Eigen::Quaterniond q_begin;
  Eigen::Quaterniond q_end;
  if (!prior_pose_at(stamp_begin, &p_begin, &q_begin, nullptr) ||
      !prior_pose_at(stamp_end, &p_end, &q_end, nullptr)) {
    return false;
  }
  const Eigen::Quaterniond q_begin_inv = q_begin.conjugate();
  *rotation = (q_begin_inv * q_end).toRotationMatrix();
  *translation = q_begin_inv * (p_end - p_begin);
  return true;
}

void reset_prior(bool enabled) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  prior.stamps.clear();
  prior.positions.clear();
  prior.orientations.clear();
  prior.covariance.setZero();
  prior.covariance_set = false;
  prior.covariance_mode_known = false;
  prior.enabled = enabled;
  prior.closed = false;
  prior.latest_ticket = 0;
  prior.retain_from = -std::numeric_limits<double>::infinity();
}

void close_prior() {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  prior.closed = true;
}

std::uint64_t push_prior_pose(double stamp,
                              const Eigen::Vector3d& position,
                              const Eigen::Quaterniond& orientation,
                              const Eigen::Matrix<double, 6, 6>* covariance) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  if (!prior.enabled) {
    throw std::logic_error("push_prior_pose requires enable_prior=True");
  }
  if (prior.closed) {
    throw std::logic_error("cannot push prior pose after finish()");
  }
  if (!std::isfinite(stamp) || !position.allFinite() || !orientation.coeffs().allFinite() ||
      !(orientation.norm() > 0.0)) {
    throw std::invalid_argument("prior pose must contain finite values and a non-zero quaternion");
  }
  if (!prior.stamps.empty() && !(stamp > prior.stamps.back())) {
    throw std::invalid_argument("prior pose stamps must be strictly increasing");
  }
  const bool first_covariance = !prior.covariance_mode_known;
  if (!first_covariance && prior.covariance_set != (covariance != nullptr)) {
    throw std::invalid_argument(
        "prior covariance must be present on every pose or absent on every pose");
  }
  if (covariance != nullptr) {
    if (!covariance->allFinite() ||
        !covariance->isApprox(covariance->transpose(), 1e-12) ||
        Eigen::LLT<Eigen::Matrix<double, 6, 6>>(*covariance).info() != Eigen::Success) {
      throw std::invalid_argument("prior covariance must be finite, symmetric, and positive definite");
    }
    // The check above already established that every earlier pose carried one.
    if (!first_covariance && !prior.covariance.isApprox(*covariance, 1e-12)) {
      throw std::invalid_argument(
          "prior covariance must stay fixed for a run; adaptive weighting is not supported");
    }
    prior.covariance = *covariance;
  }
  prior.covariance_mode_known = true;
  prior.covariance_set = covariance != nullptr;

  Eigen::Quaterniond q = orientation.normalized();
  if (!prior.orientations.empty() &&
      q.coeffs().dot(prior.orientations.back().coeffs()) < 0.0) {
    q.coeffs() *= -1.0;
  }
  prior.stamps.push_back(stamp);
  prior.positions.push_back(position);
  prior.orientations.push_back(q);

  // Trim ahead of the oldest sweep any live local-BA window can still
  // reference, keeping two predecessors: one to bracket a query at
  // `retain_from` itself, and one more so that bracket's own central-difference
  // velocity is still computable. With both, retention cannot change the value
  // of any query the estimator can still make. Never trim below the two samples
  // an interpolation needs at all.
  if (std::isfinite(prior.retain_from) && prior.stamps.size() > 2) {
    const auto keep =
        std::lower_bound(prior.stamps.begin(), prior.stamps.end(), prior.retain_from);
    std::size_t drop = static_cast<std::size_t>(keep - prior.stamps.begin());
    drop = drop > 2 ? drop - 2 : 0;
    drop = std::min(drop, prior.stamps.size() - 2);
    if (drop > 0) {
      prior.stamps.erase(prior.stamps.begin(), prior.stamps.begin() + drop);
      prior.positions.erase(prior.positions.begin(), prior.positions.begin() + drop);
      prior.orientations.erase(prior.orientations.begin(), prior.orientations.begin() + drop);
    }
  }

  const std::uint64_t ticket = ++prior.latest_ticket;
  {
    auto& rec = recorder();
    std::lock_guard<std::mutex> rec_lock(rec.mutex);
    rec.pipeline.latest_prior_ticket = ticket;
  }
  return ticket;
}

void retain_prior_from(double stamp) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  prior.retain_from = stamp;
}

bool prior_enabled() {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  return prior.enabled;
}

std::size_t prior_buffer_size() {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  return prior.stamps.size();
}

bool prior_ready_for_sweep(double stamp_begin, double stamp_end) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  if (!prior.enabled) {
    return true;
  }
  if (prior.closed) {
    return true;
  }
  if (prior.stamps.empty()) {
    return false;
  }
  // A left-edge miss can never be repaired by a strictly ordered future push;
  // let this sweep use upstream prediction. A right-edge miss is lookahead and
  // must park rather than extrapolate.
  if (stamp_begin < prior.stamps.front()) {
    return true;
  }
  // Pose interpolation needs the first sample after the sweep. Velocity at
  // that right bracket needs one more sample so its central difference is
  // final before the estimator consumes it. This makes results independent of
  // how the producer batches pushes. close_prior() finalizes the tail with the
  // same one-sided endpoint difference used by the former bulk API -- and a
  // producer whose prior is shorter than the recording has to call it, or every
  // remaining sweep parks here forever.
  const auto first_after =
      std::upper_bound(prior.stamps.begin(), prior.stamps.end(), stamp_end);
  return prior.stamps.end() - first_after >= 2;
}

bool prior_covers(double stamp_begin, double stamp_end) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  // Two samples, because covering a stamp means being able to interpolate it;
  // this is the same guard prior_pose_at() applies.
  if (prior.stamps.size() < 2 || !std::isfinite(stamp_begin) || !std::isfinite(stamp_end)) {
    return false;
  }
  return stamp_begin >= prior.stamps.front() && stamp_end <= prior.stamps.back();
}

bool prior_pose_at(double stamp,
                   Eigen::Vector3d* position,
                   Eigen::Quaterniond* orientation,
                   Eigen::Vector3d* velocity) {
  auto& prior = prior_trajectory();
  std::lock_guard<std::mutex> lock(prior.mutex);
  const std::size_t count = prior.stamps.size();
  if (count == 0 || !std::isfinite(stamp)) {
    return false;
  }
  // Outside the prior's span there is nothing to interpolate; the caller falls
  // back to upstream rather than extrapolating.
  if (count < 2 || stamp < prior.stamps.front() || stamp > prior.stamps.back()) {
    return false;
  }

  const auto upper = std::upper_bound(prior.stamps.begin(), prior.stamps.end(), stamp);
  std::size_t hi = static_cast<std::size_t>(upper - prior.stamps.begin());
  if (hi == 0) {
    hi = 1;
  }
  if (hi >= count) {
    hi = count - 1;
  }
  const std::size_t lo = hi - 1;

  const double span = prior.stamps[hi] - prior.stamps[lo];
  const double u = span > 0.0 ? (stamp - prior.stamps[lo]) / span : 0.0;
  if (position != nullptr) {
    *position = prior.positions[lo] + u * (prior.positions[hi] - prior.positions[lo]);
  }
  if (orientation != nullptr) {
    *orientation = prior.orientations[lo].slerp(u, prior.orientations[hi]);
  }
  if (velocity != nullptr) {
    const Eigen::Vector3d v_lo = prior_velocity_at(prior, lo);
    const Eigen::Vector3d v_hi = prior_velocity_at(prior, hi);
    *velocity = v_lo + u * (v_hi - v_lo);
  }
  return true;
}

}  // namespace voxelslam_offline

struct VoxelSlamOptions {
  std::string save_path = "/tmp/voxel_slam_offline/";
  std::string bagname = "offline";

  int lidar_type = VELODYNE;
  double blind = 2.8;
  int point_filter_num = 3;
  std::vector<double> extrinsic_tran = {0.0, 0.0, 0.28};
  std::vector<double> extrinsic_rota = {1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0};
  bool is_save_map = false;

  double odom_cov_gyr = 0.01;
  double odom_cov_acc = 1.0;
  double odom_rdw_gyr = 0.0001;
  double odom_rdw_acc = 0.0001;
  double down_size = 0.25;
  double beam_err = 0.01;
  double dept_err = 0.01;
  double voxel_size = 2.0;
  double min_eigen_value = 0.01;
  int degrade_bound = 100;
  bool point_notime = false;

  int win_size = 10;
  int max_layer = 2;
  double lba_cov_gyr = 0.01;
  double lba_cov_acc = 1.0;
  double lba_rdw_gyr = 0.0001;
  double lba_rdw_acc = 0.0001;
  int min_ba_point = 1;
  std::vector<double> plane_eigen_value_thre = {4.0, 4.0, 4.0, 4.0};
  double imu_coef = 0.0001;
  int thread_num = 5;

  double loop_jud_default = 0.45;
  double loop_icp_eigval = 15.0;
  double loop_ratio_drift = 0.01;
  int loop_curr_halt = 10;
  int loop_prev_halt = 10;
  int loop_acsize = 2;
  int loop_mgsize = 2;
  int loop_is_high_fly = 0;
  double loop_dwell_seconds = 0.0;

  double gba_voxel_size = 2.0;
  double gba_min_eigen_value = 0.01;
  std::vector<double> gba_eigen_value_array = {9.0, 9.0, 9.0, 9.0};
  int gba_total_max_iter = 3;

  bool emit_deskewed_points = false;
  bool enable_loop_closure = true;
  bool enable_global_mapping = true;

  // Opens the causal prior input. Data itself arrives through push_prior_pose,
  // never through constructor options.
  bool enable_prior = false;

  // The same accuracy, used as the weight of the prior's relative-pose factor
  // in the local BA. Separate from the IEKF sigmas above because the two act on
  // different estimators and are worth tuning apart.
  double prior_ba_sigma_rot = 0.0;
  double prior_ba_sigma_pos = 0.0;

  // Undistort points against the prior instead of IMU dead reckoning.
  bool prior_deskew = false;

};

struct Result {
  std::vector<voxelslam_offline::PoseRecord> poses;
  voxelslam_offline::Metrics metrics;
  std::vector<voxelslam_offline::EventRecord> events;
  std::vector<voxelslam_offline::PriorSweepRecord> prior_trace;
};

static py::array_t<double> poses_to_array(const std::vector<voxelslam_offline::PoseRecord>& poses) {
  py::array_t<double> out({static_cast<py::ssize_t>(poses.size()), static_cast<py::ssize_t>(8)});
  auto dst = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(poses.size()); ++i) {
    const auto& p = poses[static_cast<std::size_t>(i)];
    dst(i, 0) = p.stamp;
    dst(i, 1) = p.x;
    dst(i, 2) = p.y;
    dst(i, 3) = p.z;
    dst(i, 4) = p.qx;
    dst(i, 5) = p.qy;
    dst(i, 6) = p.qz;
    dst(i, 7) = p.qw;
  }
  return out;
}

// One row per sweep, in sweep order: the columns named by PRIOR_TRACE_COLUMNS
// below. A plain array rather than a list of dicts because it is a few thousand
// rows per run and the host writes it straight out as a CSV.
static const char* const PRIOR_TRACE_COLUMNS[] = {
    "stamp", "eigen_min", "eigen_mid", "eigen_max", "match_count",
    "prior_primed", "degenerate", "disagreement_rot", "disagreement_pos",
    "propagated_sigma_rot", "propagated_sigma_pos",
    "propagated_pose_velocity_correlation", "propagated_pose_bias_correlation",
};

static py::array_t<double> prior_trace_to_array(
    const std::vector<voxelslam_offline::PriorSweepRecord>& rows) {
  constexpr py::ssize_t columns =
      static_cast<py::ssize_t>(sizeof(PRIOR_TRACE_COLUMNS) / sizeof(PRIOR_TRACE_COLUMNS[0]));
  py::array_t<double> out({static_cast<py::ssize_t>(rows.size()), columns});
  auto dst = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rows.size()); ++i) {
    const auto& r = rows[static_cast<std::size_t>(i)];
    dst(i, 0) = r.stamp;
    dst(i, 1) = r.eigenvalue_min;
    dst(i, 2) = r.eigenvalue_mid;
    dst(i, 3) = r.eigenvalue_max;
    dst(i, 4) = static_cast<double>(r.match_count);
    dst(i, 5) = r.prior_primed ? 1.0 : 0.0;
    dst(i, 6) = r.degenerate ? 1.0 : 0.0;
    dst(i, 7) = r.disagreement_rot;
    dst(i, 8) = r.disagreement_pos;
    dst(i, 9) = r.propagated_sigma_rot;
    dst(i, 10) = r.propagated_sigma_pos;
    dst(i, 11) = r.propagated_pose_velocity_correlation;
    dst(i, 12) = r.propagated_pose_bias_correlation;
  }
  return out;
}

static py::array_t<double> pose_to_array(const voxelslam_offline::PoseRecord& pose) {
  py::array_t<double> out({static_cast<py::ssize_t>(8)});
  auto dst = out.mutable_unchecked<1>();
  dst(0) = pose.stamp;
  dst(1) = pose.x;
  dst(2) = pose.y;
  dst(3) = pose.z;
  dst(4) = pose.qx;
  dst(5) = pose.qy;
  dst(6) = pose.qz;
  dst(7) = pose.qw;
  return out;
}

static py::array_t<double> points_to_array(const std::vector<voxelslam_offline::PointRecord>& points) {
  py::array_t<double> out({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(5)});
  auto dst = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(points.size()); ++i) {
    const auto& p = points[static_cast<std::size_t>(i)];
    dst(i, 0) = p.stamp;
    dst(i, 1) = p.x;
    dst(i, 2) = p.y;
    dst(i, 3) = p.z;
    dst(i, 4) = p.intensity;
  }
  return out;
}

static py::list point_batches_to_list(const std::vector<std::vector<voxelslam_offline::PointRecord>>& batches) {
  py::list out;
  for (const auto& batch : batches) {
    out.append(points_to_array(batch));
  }
  return out;
}

static py::dict scalar_stats_to_dict(const voxelslam_offline::ScalarStats& stats) {
  py::dict out;
  out["count"] = stats.count;
  if (stats.count == 0) {
    out["min"] = py::none();
    out["mean"] = py::none();
    out["max"] = py::none();
    return out;
  }
  out["min"] = stats.min;
  out["mean"] = stats.sum / static_cast<double>(stats.count);
  out["max"] = stats.max;
  return out;
}

static py::dict metrics_to_dict(const voxelslam_offline::Metrics& d) {
  py::dict out;

  py::dict odometry;
  odometry["degrade_resets"] = d.odometry_degrade_resets;
  odometry["prior_applied"] = d.odometry_prior_applied;
  odometry["prior_degenerate_fallback"] = d.odometry_prior_fallback;
  odometry["iekf_eigen_min"] = scalar_stats_to_dict(d.iekf_eigenvalue_min);
  odometry["iekf_eigen_ratio"] = scalar_stats_to_dict(d.iekf_eigenvalue_ratio);
  odometry["iekf_match_count"] = scalar_stats_to_dict(d.iekf_match_count);
  odometry["prior_disagreement_rot"] = scalar_stats_to_dict(d.prior_disagreement_rot);
  odometry["prior_disagreement_pos"] = scalar_stats_to_dict(d.prior_disagreement_pos);
  odometry["propagated_sigma_rot"] = scalar_stats_to_dict(d.propagated_sigma_rot);
  odometry["propagated_sigma_pos"] = scalar_stats_to_dict(d.propagated_sigma_pos);
  odometry["propagated_pose_velocity_correlation"] =
      scalar_stats_to_dict(d.propagated_pose_velocity_correlation);
  odometry["propagated_pose_bias_correlation"] =
      scalar_stats_to_dict(d.propagated_pose_bias_correlation);
  out["odometry"] = odometry;

  py::dict loop;
  loop["candidates"] = d.loop_candidates;
  loop["score_passed"] = d.loop_score_passed;
  loop["icp_converged"] = d.loop_icp_converged;
  loop["icp_passed"] = d.loop_icp_passed;
  loop["icp_failed"] = d.loop_icp_failed;
  loop["drift_passed"] = d.loop_drift_passed;
  loop["drift_rejected"] = d.loop_drift_rejected;
  loop["edges_added"] = d.loop_edges_added;
  loop["graph_optimizations"] = d.loop_graph_optimizations;
  loop["updates_applied"] = d.loop_updates_applied;
  loop["score"] = scalar_stats_to_dict(d.loop_score);
  loop["icp_match_count"] = scalar_stats_to_dict(d.loop_icp_match_count);
  loop["icp_eigen_min"] = scalar_stats_to_dict(d.loop_icp_eigen_min);
  loop["icp_eigen_mid"] = scalar_stats_to_dict(d.loop_icp_eigen_mid);
  loop["icp_eigen_max"] = scalar_stats_to_dict(d.loop_icp_eigen_max);
  loop["drift_ratio"] = scalar_stats_to_dict(d.loop_drift_ratio);
  loop["graph_pose_count"] = scalar_stats_to_dict(d.loop_graph_pose_count);
  out["loop_closure"] = loop;

  py::dict gba;
  gba["started"] = d.gba_started;
  gba["completed"] = d.gba_completed;
  gba["keyframes"] = scalar_stats_to_dict(d.gba_keyframes);
  gba["runtime_seconds"] = scalar_stats_to_dict(d.gba_runtime_seconds);
  gba["pose_count"] = scalar_stats_to_dict(d.gba_pose_count);
  gba["stage1_edges"] = scalar_stats_to_dict(d.gba_stage1_edges);
  gba["stage2_edges"] = scalar_stats_to_dict(d.gba_stage2_edges);
  out["global_ba"] = gba;

  return out;
}

static py::list events_to_list(const std::vector<voxelslam_offline::EventRecord>& events) {
  py::list out;
  for (const auto& event : events) {
    py::dict item;
    item["type"] = event.type;
    item["stamp"] = event.stamp;
    item["lidar_ticket"] = event.lidar_ticket;
    item["reason"] = event.reason;
    out.append(item);
  }
  return out;
}

static py::dict status_to_dict(const voxelslam_offline::PipelineStatus& status,
                               std::size_t pending_imu,
                               std::size_t buffered_prior,
                               std::size_t pending_lidar,
                               std::size_t pending_loop_scanposes,
                               bool loop_processing,
                               bool loop_update_pending,
                               bool loop_reset_pending,
                               bool loop_enabled,
                               bool gba_enabled,
                               bool gba_idle) {
  py::dict out;

  py::dict imu;
  imu["latest_ticket"] = status.latest_imu_ticket;
  imu["pending_queue"] = pending_imu;
  out["imu"] = imu;

  py::dict prior;
  prior["enabled"] = voxelslam_offline::prior_enabled();
  prior["latest_ticket"] = status.latest_prior_ticket;
  prior["buffered_poses"] = buffered_prior;
  out["prior"] = prior;

  py::dict lidar;
  lidar["latest_ticket"] = status.latest_lidar_ticket;
  lidar["latest_processed_ticket"] = status.latest_lidar_processed_ticket;
  lidar["completed"] = status.lidar_completed;
  lidar["completed_with_pose"] = status.lidar_completed_with_pose;
  lidar["completed_without_pose"] = status.lidar_completed_without_pose;
  lidar["pending_queue"] = pending_lidar;
  // Sweeps accepted but never given a terminal accounting state: whatever the
  // estimator could not cover with the data submitted. A recording whose tail
  // has no trailing IMU always ends with a few; that is a property of the
  // recording, so this is reported and never fatal.
  lidar["uncovered"] = status.latest_lidar_ticket > status.lidar_completed
                           ? status.latest_lidar_ticket - status.lidar_completed
                           : std::uint64_t{0};
  out["lidar"] = lidar;

  py::dict odometry;
  odometry["waiting_for_imu"] = status.odometry_waiting_for_imu;
  odometry["waiting_lidar_ticket"] = status.odometry_waiting_lidar_ticket;
  odometry["waiting_scan_end_time"] = status.odometry_waiting_scan_end_time;
  odometry["waiting_for_prior"] = status.odometry_waiting_for_prior;
  odometry["waiting_prior_begin_time"] = status.odometry_waiting_prior_begin_time;
  odometry["waiting_prior_end_time"] = status.odometry_waiting_prior_end_time;
  out["odometry"] = odometry;

  py::dict loop;
  loop["pending_queue"] = pending_loop_scanposes;
  loop["processing"] = loop_processing;
  loop["update_pending"] = loop_update_pending;
  loop["reset_pending"] = loop_reset_pending;
  out["loop"] = loop;

  py::dict workers;
  workers["odometry_enabled"] = true;
  workers["loop_enabled"] = loop_enabled;
  workers["gba_enabled"] = gba_enabled;
  workers["odometry_finished"] = status.odometry_thread_finished;
  workers["loop_finished"] = !loop_enabled || status.loop_thread_finished;
  workers["gba_finished"] = !gba_enabled || status.gba_thread_finished;
  // Drained, not merely between passes -- see workers_idle().
  workers["gba_idle"] = !gba_enabled || gba_idle;
  out["workers"] = workers;

  return out;
}

class VoxelSlam {
 public:
  explicit VoxelSlam(const VoxelSlamOptions& options) : options_(options) {
    validate_options(options_);
    if (options_.is_save_map) {
      std::filesystem::create_directories(options_.save_path);
    }
    reset_upstream_buffers();
    voxelslam_offline::reset_records(options_.emit_deskewed_points);
    voxelslam_offline::reset_prior(options_.enable_prior);
    voxelslam_offline::set_prior_ba_sigma(options_.prior_ba_sigma_rot,
                                          options_.prior_ba_sigma_pos);
    voxelslam_offline::set_prior_deskew(options_.prior_deskew);
    configure_node(options_);

    slam_ = std::make_unique<VOXEL_SLAM>(node_);
    reset_window_permutation();

    if (options_.enable_loop_closure) {
      loop_thread_ = std::thread([this]() { run_loop(); });
    }
    if (options_.enable_global_mapping) {
      gba_thread_ = std::thread([this]() { run_gba(); });
    }
    odom_thread_ = std::thread([this]() { run_odometry(); });
  }

  ~VoxelSlam() {
    if (!finished_) {
      // Teardown of a session the host abandoned -- an exception in its replay
      // loop, or a dropped reference. Join the workers, but do not wait for
      // pending work: its results are already unreachable, and the wait a
      // completed run performs is unbounded by design, which a destructor
      // cannot be. finish() is the only path that produces a result.
      try {
        join_workers();
        finished_ = true;
      } catch (const std::exception& exc) {
        std::fprintf(stderr, "VoxelSlam cleanup failed: %s\n", exc.what());
      } catch (...) {
        std::fprintf(stderr, "VoxelSlam cleanup failed with an unknown exception\n");
      }
    }
  }

  std::uint64_t push_imu(double stamp,
                         const std::vector<double>& linear_acceleration,
                         const std::vector<double>& angular_velocity) {
    if (finished_) {
      throw std::runtime_error("cannot push IMU after finish()");
    }
    if (linear_acceleration.size() != 3 || angular_velocity.size() != 3) {
      throw std::invalid_argument("linear_acceleration and angular_velocity must have length 3");
    }

    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu());
    msg->header.stamp.fromSec(stamp);
    msg->linear_acceleration.x = linear_acceleration[0];
    msg->linear_acceleration.y = linear_acceleration[1];
    msg->linear_acceleration.z = linear_acceleration[2];
    msg->angular_velocity.x = angular_velocity[0];
    msg->angular_velocity.y = angular_velocity[1];
    msg->angular_velocity.z = angular_velocity[2];

    const std::uint64_t ticket = ++latest_imu_ticket_;
    voxelslam_offline::record_imu_pushed(ticket);
    std::lock_guard<std::mutex> lock(mBuf);
    imu_last_time = stamp;
    imu_buf.push_back(msg);
    return ticket;
  }

  std::uint64_t push_prior_pose(double stamp,
                                const std::vector<double>& position,
                                const std::vector<double>& orientation,
                                py::object covariance_obj = py::none()) {
    if (finished_) {
      throw std::runtime_error("cannot push prior pose after finish()");
    }
    if (position.size() != 3 || orientation.size() != 4) {
      throw std::invalid_argument("position and orientation must have lengths 3 and 4");
    }
    Eigen::Matrix<double, 6, 6> covariance;
    const Eigen::Matrix<double, 6, 6>* covariance_ptr = nullptr;
    if (!covariance_obj.is_none()) {
      py::array_t<double, py::array::c_style | py::array::forcecast> array =
          py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
              covariance_obj);
      if (array.ndim() != 2 || array.shape(0) != 6 || array.shape(1) != 6) {
        throw std::invalid_argument("covariance must have shape (6, 6)");
      }
      const auto values = array.unchecked<2>();
      for (py::ssize_t row = 0; row < 6; ++row) {
        for (py::ssize_t col = 0; col < 6; ++col) {
          covariance(row, col) = values(row, col);
        }
      }
      covariance_ptr = &covariance;
    }
    return voxelslam_offline::push_prior_pose(
        stamp,
        Eigen::Vector3d(position[0], position[1], position[2]),
        Eigen::Quaterniond(orientation[3], orientation[0], orientation[1], orientation[2]),
        covariance_ptr);
  }

  // Declare the prior stream complete without ending the run. Sweeps past the
  // prior's last sample then fall back to upstream prediction immediately
  // instead of parking for lookahead that cannot arrive -- which is what keeps
  // synchronize() a real barrier, and the LiDAR queue bounded, for the rest of
  // a run whose prior is shorter than its recording. Idempotent.
  void close_prior() {
    if (finished_) {
      return;
    }
    voxelslam_offline::close_prior();
  }

  std::uint64_t push_lidar(double stamp,
                           py::array_t<float, py::array::c_style | py::array::forcecast> points,
                           py::object relative_times_obj = py::none(),
                           py::object intensities_obj = py::none(),
                           double scan_duration = -1.0,
                           bool stamp_is_end = false) {
    if (finished_) {
      throw std::runtime_error("cannot push lidar after finish()");
    }

    const auto pts = points.unchecked<2>();
    if (pts.shape(1) != 3) {
      throw std::invalid_argument("points must be an Nx3 array");
    }
    const py::ssize_t count = pts.shape(0);

    std::vector<float> relative_times;
    if (!relative_times_obj.is_none()) {
      py::array_t<float, py::array::c_style | py::array::forcecast> relative_times_arr =
          py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(relative_times_obj);
      const auto times = relative_times_arr.unchecked<1>();
      if (times.shape(0) != count) {
        throw std::invalid_argument("relative_times must have length N");
      }
      relative_times.resize(static_cast<std::size_t>(count));
      for (py::ssize_t i = 0; i < count; ++i) {
        relative_times[static_cast<std::size_t>(i)] = times(i);
      }
    } else if (scan_duration > 0.0 && count > 1) {
      relative_times.resize(static_cast<std::size_t>(count));
      for (py::ssize_t i = 0; i < count; ++i) {
        relative_times[static_cast<std::size_t>(i)] =
            static_cast<float>(scan_duration * static_cast<double>(i) / static_cast<double>(count - 1));
      }
    }

    std::vector<float> intensities;
    if (!intensities_obj.is_none()) {
      py::array_t<float, py::array::c_style | py::array::forcecast> intensities_arr =
          py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(intensities_obj);
      const auto ints = intensities_arr.unchecked<1>();
      if (ints.shape(0) != count) {
        throw std::invalid_argument("intensities must have length N");
      }
      intensities.resize(static_cast<std::size_t>(count));
      for (py::ssize_t i = 0; i < count; ++i) {
        intensities[static_cast<std::size_t>(i)] = ints(i);
      }
    }

    // Every valid submission receives a ticket and a terminal accounting state,
    // even when upstream-compatible preprocessing removes every point.
    const std::uint64_t ticket = ++latest_lidar_ticket_;
    voxelslam_offline::record_lidar_pushed(ticket);

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    cloud->reserve(static_cast<std::size_t>(count));
    for (py::ssize_t i = 0; i < count; ++i) {
      if (options_.point_filter_num > 1 && (i % options_.point_filter_num) != 0) {
        continue;
      }

      PointType point;
      point.x = pts(i, 0);
      point.y = pts(i, 1);
      point.z = pts(i, 2);
      point.intensity = intensities.empty() ? 0.0f : intensities[static_cast<std::size_t>(i)];
      point.curvature = relative_times.empty() ? 0.0f : relative_times[static_cast<std::size_t>(i)];

      const float range2 = point.x * point.x + point.y * point.y + point.z * point.z;
      if (range2 > static_cast<float>(options_.blind * options_.blind)) {
        cloud->push_back(point);
      }
    }

    if (cloud->empty()) {
      std::fprintf(stderr, "lidar sweep at %.9f is empty after filtering\n", stamp);
      voxelslam_offline::record_lidar_processed(ticket, false);
      return ticket;
    }

    std::sort(cloud->begin(), cloud->end(), [](const PointType& a, const PointType& b) {
      return a.curvature < b.curvature;
    });
    if (scan_duration > 0.0) {
      while (!cloud->empty() && cloud->back().curvature > scan_duration * 1.1) {
        cloud->points.pop_back();
      }
    }
    if (cloud->empty()) {
      std::fprintf(stderr, "lidar sweep at %.9f has no points within scan_duration\n", stamp);
      voxelslam_offline::record_lidar_processed(ticket, false);
      return ticket;
    }
    const double begin_stamp = stamp_is_end ? stamp - cloud->back().curvature : stamp;

    std::lock_guard<std::mutex> lock(mBuf);
    time_buf.push_back(begin_stamp);
    pcl_buf.push_back(cloud);
    lidar_ticket_buf.push_back(ticket);
    lidar_stamps_.emplace_back(ticket, begin_stamp);
    const std::uint64_t processed =
        voxelslam_offline::snapshot_pipeline_status().latest_lidar_processed_ticket;
    const std::uint64_t history = static_cast<std::uint64_t>(options_.win_size + 2);
    const std::uint64_t discard_through = processed > history ? processed - history : 0;
    while (!lidar_stamps_.empty() && lidar_stamps_.front().first <= discard_through) {
      lidar_stamps_.pop_front();
    }
    if (options_.enable_prior && !lidar_stamps_.empty()) {
      voxelslam_offline::retain_prior_from(lidar_stamps_.front().second);
    }
    return ticket;
  }

  const Result& finish() {
    if (finished_) {
      return result_;
    }

    // End-of-stream makes an uncovered right edge definitive: parked sweeps
    // can now fall back to upstream instead of waiting for impossible lookahead.
    voxelslam_offline::close_prior();
    std::exception_ptr finish_error;
    try {
      wait_for_processed(latest_lidar_ticket_);
    } catch (...) {
      finish_error = std::current_exception();
    }
    join_workers();

    result_.poses = voxelslam_offline::take_poses();
    result_.metrics = voxelslam_offline::snapshot_metrics();
    result_.events = voxelslam_offline::snapshot_events();
    result_.prior_trace = voxelslam_offline::prior_sweep_trace();
    finished_ = true;
    if (finish_error) {
      std::rethrow_exception(finish_error);
    }
    throw_if_thread_error();
    return result_;
  }

  // Stop the workers and join them. Whatever they have already accepted is
  // still finished; nothing new is admitted.
  void join_workers() {
    request_finish();
    if (has_thread_error()) {
      node_.setParam("__shutdown", true);
    }

    if (odom_thread_.joinable()) {
      odom_thread_.join();
    }
    if (loop_thread_.joinable()) {
      loop_thread_.join();
    }

    node_.setParam("__shutdown", true);
    if (gba_thread_.joinable()) {
      gba_thread_.join();
    }
  }

  void request_finish() {
    if (slam_) {
      slam_->is_finish = true;
    }
    node_.setParam("finish", true);
  }

  bool is_finished() const {
    return finished_;
  }

  py::object latest_pose() const {
    throw_if_thread_error();
    const auto poses = current_poses();
    if (poses.empty()) {
      return py::none();
    }
    return pose_to_array(poses.back());
  }

  py::array_t<double> trajectory() const {
    throw_if_thread_error();
    return poses_to_array(current_poses());
  }

  py::dict metrics() const {
    throw_if_thread_error();
    return metrics_to_dict(voxelslam_offline::snapshot_metrics());
  }

  py::dict status() const {
    throw_if_thread_error();
    return current_status();
  }

  std::uint64_t latest_imu_ticket() const {
    return latest_imu_ticket_;
  }

  std::uint64_t latest_prior_ticket() const {
    return voxelslam_offline::snapshot_pipeline_status().latest_prior_ticket;
  }

  std::uint64_t latest_lidar_ticket() const {
    return latest_lidar_ticket_;
  }

  // The replay barrier. It blocks until the workers have made all the progress
  // they can with the data submitted so far. That is either the target ticket
  // completing, or full quiescence: every worker drained and the estimator
  // parked on a sweep it cannot advance without input the host has not
  // submitted. There is no deadline; a deadline would make the barrier -- and
  // with it determinism -- optional, and would turn a slow machine into a
  // different trajectory. The quiescent exit is not a deadline and not
  // optional: it is a property of the submitted data, so it holds at every
  // instant after it first holds, and the host cannot reach it early by
  // running fast.
  void wait_for_processed(std::uint64_t ticket = 0) const {
    throw_if_thread_error();
    const std::uint64_t target = ticket == 0 ? latest_lidar_ticket_ : ticket;
    if (target == 0) {
      return;
    }
    while (true) {
      throw_if_thread_error();
      const auto status = voxelslam_offline::snapshot_pipeline_status();
      if (status.latest_lidar_processed_ticket >= target && workers_idle()) {
        return;
      }
      // Order matters and must not be rearranged: a parked estimator is what
      // guarantees no further work can be queued for the loop and global-
      // mapping threads, so their drained state is only meaningful once read
      // after the park has been established. workers_idle() therefore comes
      // second, exactly as it does above.
      if (blocked_on_unsubmitted_imu(status, target) && workers_idle()) {
        return;
      }
      if (blocked_on_unsubmitted_prior(status, target) && workers_idle()) {
        return;
      }
      if (status.odometry_thread_finished &&
          status.latest_lidar_processed_ticket < target) {
        throw std::runtime_error(
            "VoxelSLAM odometry worker exited before processing lidar ticket " +
            std::to_string(target) + " (latest processed ticket " +
            std::to_string(status.latest_lidar_processed_ticket) + ")");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void synchronize(std::uint64_t ticket = 0) const {
    wait_for_processed(ticket);
  }

  py::list pop_deskewed_scans() const {
    throw_if_thread_error();
    return point_batches_to_list(voxelslam_offline::take_deskewed_scans());
  }

 private:
  void run_odometry() {
    try {
      slam_->thd_odometry_localmapping(node_);
    } catch (...) {
      capture_thread_exception();
    }
    voxelslam_offline::record_odometry_thread_finished();
  }

  void run_loop() {
    try {
      slam_->thd_loop_closure(node_);
    } catch (...) {
      capture_thread_exception();
    }
    voxelslam_offline::record_loop_thread_finished();
  }

  void run_gba() {
    try {
      slam_->thd_globalmapping(node_);
    } catch (...) {
      capture_thread_exception();
    }
    voxelslam_offline::record_gba_thread_finished();
  }

  void capture_thread_exception() {
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      if (!thread_error_) {
        thread_error_ = std::current_exception();
      }
    }
    if (slam_) {
      slam_->is_finish = true;
      // Wake a loop thread waiting for final GBA without letting it consume a
      // partial result from a worker that failed.
      slam_->gba_cancelled = true;
      slam_->gba_flag = false;
    }
    node_.setParam("finish", true);
    node_.setParam("__shutdown", true);
  }

  bool has_thread_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return static_cast<bool>(thread_error_);
  }

  void throw_if_thread_error() const {
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error = thread_error_;
    }
    if (error) {
      std::rethrow_exception(error);
    }
  }

  // True when the estimator has parked a sweep at or before `target` because the
  // IMU submitted so far does not extend past that sweep's end, so no amount of
  // waiting can advance it.
  //
  // The predicate is re-evaluated here against the IMU horizon as it stands
  // now, not against the worker's opinion at the moment it published the park.
  // That is what makes a stale park record harmless: to leave a parked sweep
  // the worker needs `imu_last_time` strictly past that sweep's end time, and
  // `imu_last_time` never decreases, so any park the worker has already left
  // reads as unblocked here. The converse -- reporting blocked while the worker
  // is mid-sweep -- cannot happen either, because the flag is cleared in the
  // same call that lets the sweep through, before any estimator work runs.
  //
  // The host is the only producer of IMU and it is inside this call, so the
  // horizon cannot move while the barrier is deciding. The result is a function
  // of the submitted data alone; thread timing cannot change it, only how long
  // the loop spins before observing it.
  bool blocked_on_unsubmitted_imu(const voxelslam_offline::PipelineStatus& status,
                                  std::uint64_t target) const {
    if (!status.odometry_waiting_for_imu ||
        status.odometry_waiting_lidar_ticket == 0 ||
        status.odometry_waiting_lidar_ticket > target) {
      return false;
    }
    // Sweeps leave the queue in order, so a sweep parked at or before `target`
    // blocks `target` itself and everything queued behind it.
    double submitted_imu_end = 0.0;
    {
      std::lock_guard<std::mutex> lock(mBuf);
      submitted_imu_end = imu_last_time;
    }
    return submitted_imu_end <= status.odometry_waiting_scan_end_time;
  }

  // The prior's counterpart to blocked_on_unsubmitted_imu(), and it re-evaluates
  // the worker's own predicate against the prior as it stands now rather than a
  // second rule that happens to agree. A stale park is harmless for the same
  // reason: the buffer only ever gains samples on the right, so a sweep the
  // worker has already left reads as ready here.
  bool blocked_on_unsubmitted_prior(const voxelslam_offline::PipelineStatus& status,
                                    std::uint64_t target) const {
    return status.odometry_waiting_for_prior &&
           status.odometry_waiting_lidar_ticket != 0 &&
           status.odometry_waiting_lidar_ticket <= target &&
           !voxelslam_offline::prior_ready_for_sweep(
               status.odometry_waiting_prior_begin_time,
               status.odometry_waiting_prior_end_time);
  }

  // True once every background worker has drained the work the sweeps fed so far.
  // The order of the tests matters and must not be rearranged: an idle loop thread
  // is what guarantees no further keyframe can be queued, so the global-bundle-
  // adjustment thread's drained flag is only meaningful once it has been read after
  // that. Reading the flag first would admit a keyframe queued and consumed in
  // between and report both threads idle while a bottom-up pass was still due.
  bool workers_idle() const {
    if (!slam_) {
      return true;
    }
    if (options_.enable_loop_closure) {
      std::lock_guard<std::mutex> lock(slam_->mtx_loop);
      if (!(slam_->buf_lba2loop.empty() &&
            slam_->loop_processing == 0 &&
            slam_->loop_detect == 0 &&
            slam_->reset_flag == 0)) {
        return false;
      }
    }
    // Drained, not merely between passes: the flag is cleared by whoever queues the
    // work and re-set only when that thread finds its input empty, so a pass that
    // has not started yet still reads as busy.
    return !options_.enable_global_mapping || slam_->gba_quiescent != 0;
  }

  std::vector<voxelslam_offline::PoseRecord> current_poses() const {
    if (finished_) {
      return result_.poses;
    }
    return voxelslam_offline::snapshot_best_poses();
  }

  py::dict current_status() const {
    std::size_t pending_imu = 0;
    std::size_t pending_lidar = 0;
    {
      std::lock_guard<std::mutex> lock(mBuf);
      pending_imu = imu_buf.size();
      pending_lidar = pcl_buf.size();
    }

    std::size_t pending_loop_scanposes = 0;
    bool gba_idle = true;
    bool loop_processing = false;
    bool loop_update_pending = false;
    bool loop_reset_pending = false;
    if (slam_) {
      {
        std::lock_guard<std::mutex> lock(slam_->mtx_loop);
        pending_loop_scanposes = slam_->buf_lba2loop.size();
        loop_processing = slam_->loop_processing != 0;
        loop_update_pending = slam_->loop_detect != 0;
        loop_reset_pending = slam_->reset_flag != 0;
      }
      // Sampled after the loop-thread state, so a caller polling this dict sees the
      // same ordering workers_idle() relies on.
      gba_idle = finished_ || slam_->gba_quiescent != 0;
    }

    return status_to_dict(voxelslam_offline::snapshot_pipeline_status(),
                          pending_imu,
                          voxelslam_offline::prior_buffer_size(),
                          pending_lidar,
                          pending_loop_scanposes,
                          loop_processing,
                          loop_update_pending,
                          loop_reset_pending,
                          options_.enable_loop_closure,
                          options_.enable_global_mapping,
                          gba_idle);
  }

  void configure_node(const VoxelSlamOptions& o) {
    node_.setParam<std::string>("General/lid_topic", "/offline/lidar");
    node_.setParam<std::string>("General/imu_topic", "/offline/imu");
    node_.setParam<std::string>("General/bagname", o.bagname);
    node_.setParam<std::string>("General/save_path", o.save_path);
    node_.setParam<std::string>("General/previous_map", "");
    node_.setParam<int>("General/lidar_type", o.lidar_type);
    node_.setParam<double>("General/blind", o.blind);
    node_.setParam<int>("General/point_filter_num", o.point_filter_num);
    node_.setParam<std::vector<double>>("General/extrinsic_tran", o.extrinsic_tran);
    node_.setParam<std::vector<double>>("General/extrinsic_rota", o.extrinsic_rota);
    node_.setParam<int>("General/is_save_map", o.is_save_map ? 1 : 0);

    node_.setParam<double>("Odometry/cov_gyr", o.odom_cov_gyr);
    node_.setParam<double>("Odometry/cov_acc", o.odom_cov_acc);
    node_.setParam<double>("Odometry/rdw_gyr", o.odom_rdw_gyr);
    node_.setParam<double>("Odometry/rdw_acc", o.odom_rdw_acc);
    node_.setParam<double>("Odometry/down_size", o.down_size);
    node_.setParam<double>("Odometry/dept_err", o.dept_err);
    node_.setParam<double>("Odometry/beam_err", o.beam_err);
    node_.setParam<double>("Odometry/voxel_size", o.voxel_size);
    node_.setParam<double>("Odometry/min_eigen_value", o.min_eigen_value);
    node_.setParam<int>("Odometry/degrade_bound", o.degrade_bound);
    node_.setParam<int>("Odometry/point_notime", o.point_notime ? 1 : 0);

    node_.setParam<int>("LocalBA/win_size", o.win_size);
    node_.setParam<int>("LocalBA/max_layer", o.max_layer);
    node_.setParam<double>("LocalBA/cov_gyr", o.lba_cov_gyr);
    node_.setParam<double>("LocalBA/cov_acc", o.lba_cov_acc);
    node_.setParam<double>("LocalBA/rdw_gyr", o.lba_rdw_gyr);
    node_.setParam<double>("LocalBA/rdw_acc", o.lba_rdw_acc);
    node_.setParam<int>("LocalBA/min_ba_point", o.min_ba_point);
    node_.setParam<std::vector<double>>("LocalBA/plane_eigen_value_thre", o.plane_eigen_value_thre);
    node_.setParam<double>("LocalBA/imu_coef", o.imu_coef);
    node_.setParam<int>("LocalBA/thread_num", o.thread_num);

    node_.setParam<double>("Loop/jud_default", o.loop_jud_default);
    node_.setParam<double>("Loop/icp_eigval", o.loop_icp_eigval);
    node_.setParam<double>("Loop/ratio_drift", o.loop_ratio_drift);
    node_.setParam<int>("Loop/curr_halt", o.loop_curr_halt);
    node_.setParam<int>("Loop/prev_halt", o.loop_prev_halt);
    node_.setParam<int>("Loop/acsize", o.loop_acsize);
    node_.setParam<int>("Loop/mgsize", o.loop_mgsize);
    node_.setParam<int>("Loop/isHighFly", o.loop_is_high_fly);
    node_.setParam<double>("Loop/dwell_seconds", o.loop_dwell_seconds);

    node_.setParam<double>("GBA/voxel_size", o.gba_voxel_size);
    node_.setParam<double>("GBA/min_eigen_value", o.gba_min_eigen_value);
    node_.setParam<std::vector<double>>("GBA/eigen_value_array", o.gba_eigen_value_array);
    node_.setParam<int>("GBA/total_max_iter", o.gba_total_max_iter);

    node_.setParam<bool>("finish", false);
    node_.setParam<bool>("__shutdown", false);
  }

  static void validate_options(const VoxelSlamOptions& o) {
    if (o.extrinsic_tran.size() != 3) {
      throw std::invalid_argument("extrinsic_tran must have length 3");
    }
    if (o.extrinsic_rota.size() != 9) {
      throw std::invalid_argument("extrinsic_rota must have length 9");
    }
    if (o.point_filter_num <= 0) {
      throw std::invalid_argument("point_filter_num must be positive");
    }
    if (o.win_size <= 1) {
      throw std::invalid_argument("win_size must be greater than 1");
    }
    if (o.enable_loop_closure && !o.enable_global_mapping) {
      throw std::invalid_argument("enable_loop_closure requires enable_global_mapping");
    }
    if (!o.enable_prior &&
        (o.prior_ba_sigma_rot > 0.0 || o.prior_ba_sigma_pos > 0.0 || o.prior_deskew)) {
      throw std::invalid_argument("prior BA/deskew options require enable_prior");
    }
    if (o.prior_ba_sigma_rot < 0.0 || o.prior_ba_sigma_pos < 0.0 ||
        ((o.prior_ba_sigma_rot > 0.0) != (o.prior_ba_sigma_pos > 0.0))) {
      throw std::invalid_argument("prior BA sigma values must be non-negative and set together");
    }
  }

  static void reset_upstream_buffers() {
    std::lock_guard<std::mutex> lock(mBuf);
    imu_buf.clear();
    pcl_buf.clear();
    time_buf.clear();
    lidar_ticket_buf.clear();
    imu_last_time = -1.0;
    last_pcl_time = -1.0;
    current_lidar_ticket = 0;
    pl_ready = false;
    point_notime = 0;
  }

  void reset_window_permutation() {
    delete[] mp;
    mp = new int[slam_->win_size];
    for (int i = 0; i < slam_->win_size; ++i) {
      mp[i] = i;
    }
  }

  VoxelSlamOptions options_;
  ros::NodeHandle node_;
  std::unique_ptr<VOXEL_SLAM> slam_;
  std::uint64_t latest_imu_ticket_ = 0;
  std::uint64_t latest_lidar_ticket_ = 0;
  std::deque<std::pair<std::uint64_t, double>> lidar_stamps_;
  std::thread odom_thread_;
  std::thread loop_thread_;
  std::thread gba_thread_;
  mutable std::mutex error_mutex_;
  std::exception_ptr thread_error_;
  bool finished_ = false;
  Result result_;
};

PYBIND11_MODULE(_core, m) {
  py::class_<VoxelSlamOptions>(m, "VoxelSlamOptions")
      .def(py::init<>())
      .def_readwrite("save_path", &VoxelSlamOptions::save_path)
      .def_readwrite("bagname", &VoxelSlamOptions::bagname)
      .def_readwrite("lidar_type", &VoxelSlamOptions::lidar_type)
      .def_readwrite("blind", &VoxelSlamOptions::blind)
      .def_readwrite("point_filter_num", &VoxelSlamOptions::point_filter_num)
      .def_readwrite("extrinsic_tran", &VoxelSlamOptions::extrinsic_tran)
      .def_readwrite("extrinsic_rota", &VoxelSlamOptions::extrinsic_rota)
      .def_readwrite("is_save_map", &VoxelSlamOptions::is_save_map)
      .def_readwrite("odom_cov_gyr", &VoxelSlamOptions::odom_cov_gyr)
      .def_readwrite("odom_cov_acc", &VoxelSlamOptions::odom_cov_acc)
      .def_readwrite("odom_rdw_gyr", &VoxelSlamOptions::odom_rdw_gyr)
      .def_readwrite("odom_rdw_acc", &VoxelSlamOptions::odom_rdw_acc)
      .def_readwrite("down_size", &VoxelSlamOptions::down_size)
      .def_readwrite("beam_err", &VoxelSlamOptions::beam_err)
      .def_readwrite("dept_err", &VoxelSlamOptions::dept_err)
      .def_readwrite("voxel_size", &VoxelSlamOptions::voxel_size)
      .def_readwrite("min_eigen_value", &VoxelSlamOptions::min_eigen_value)
      .def_readwrite("degrade_bound", &VoxelSlamOptions::degrade_bound)
      .def_readwrite("point_notime", &VoxelSlamOptions::point_notime)
      .def_readwrite("win_size", &VoxelSlamOptions::win_size)
      .def_readwrite("max_layer", &VoxelSlamOptions::max_layer)
      .def_readwrite("lba_cov_gyr", &VoxelSlamOptions::lba_cov_gyr)
      .def_readwrite("lba_cov_acc", &VoxelSlamOptions::lba_cov_acc)
      .def_readwrite("lba_rdw_gyr", &VoxelSlamOptions::lba_rdw_gyr)
      .def_readwrite("lba_rdw_acc", &VoxelSlamOptions::lba_rdw_acc)
      .def_readwrite("min_ba_point", &VoxelSlamOptions::min_ba_point)
      .def_readwrite("plane_eigen_value_thre", &VoxelSlamOptions::plane_eigen_value_thre)
      .def_readwrite("imu_coef", &VoxelSlamOptions::imu_coef)
      .def_readwrite("thread_num", &VoxelSlamOptions::thread_num)
      .def_readwrite("loop_jud_default", &VoxelSlamOptions::loop_jud_default)
      .def_readwrite("loop_icp_eigval", &VoxelSlamOptions::loop_icp_eigval)
      .def_readwrite("loop_ratio_drift", &VoxelSlamOptions::loop_ratio_drift)
      .def_readwrite("loop_curr_halt", &VoxelSlamOptions::loop_curr_halt)
      .def_readwrite("loop_prev_halt", &VoxelSlamOptions::loop_prev_halt)
      .def_readwrite("loop_acsize", &VoxelSlamOptions::loop_acsize)
      .def_readwrite("loop_mgsize", &VoxelSlamOptions::loop_mgsize)
      .def_readwrite("loop_is_high_fly", &VoxelSlamOptions::loop_is_high_fly)
      .def_readwrite("loop_dwell_seconds", &VoxelSlamOptions::loop_dwell_seconds)
      .def_readwrite("gba_voxel_size", &VoxelSlamOptions::gba_voxel_size)
      .def_readwrite("gba_min_eigen_value", &VoxelSlamOptions::gba_min_eigen_value)
      .def_readwrite("gba_eigen_value_array", &VoxelSlamOptions::gba_eigen_value_array)
      .def_readwrite("gba_total_max_iter", &VoxelSlamOptions::gba_total_max_iter)
      .def_readwrite("emit_deskewed_points", &VoxelSlamOptions::emit_deskewed_points)
      .def_readwrite("enable_loop_closure", &VoxelSlamOptions::enable_loop_closure)
      .def_readwrite("enable_global_mapping", &VoxelSlamOptions::enable_global_mapping)
      .def_readwrite("enable_prior", &VoxelSlamOptions::enable_prior)
      .def_readwrite("prior_ba_sigma_rot", &VoxelSlamOptions::prior_ba_sigma_rot)
      .def_readwrite("prior_ba_sigma_pos", &VoxelSlamOptions::prior_ba_sigma_pos)
      .def_readwrite("prior_deskew", &VoxelSlamOptions::prior_deskew);

  py::class_<Result>(m, "Result")
      .def_property_readonly("trajectory", [](const Result& result) {
        return poses_to_array(result.poses);
      })
      .def_property_readonly("metrics", [](const Result& result) {
        return metrics_to_dict(result.metrics);
      })
      .def_property_readonly("events", [](const Result& result) {
        return events_to_list(result.events);
      })
      .def_property_readonly("prior_trace", [](const Result& result) {
        return prior_trace_to_array(result.prior_trace);
      })
      .def_property_readonly_static("prior_trace_columns", [](py::object) {
        py::list out;
        for (const char* name : PRIOR_TRACE_COLUMNS) {
          out.append(py::str(name));
        }
        return out;
      });

  py::class_<VoxelSlam>(m, "VoxelSlam")
      .def(py::init<const VoxelSlamOptions&>(), py::arg("options") = VoxelSlamOptions())
      .def("push_imu", &VoxelSlam::push_imu,
           py::arg("stamp"),
           py::arg("linear_acceleration"),
           py::arg("angular_velocity"))
      .def("push_prior_pose", &VoxelSlam::push_prior_pose,
           py::arg("stamp"),
           py::arg("position"),
           py::arg("orientation"),
           py::arg("covariance") = py::none())
      .def("close_prior", &VoxelSlam::close_prior)
      .def("push_lidar", &VoxelSlam::push_lidar,
           py::arg("stamp"),
           py::arg("points"),
           py::arg("relative_times") = py::none(),
           py::arg("intensities") = py::none(),
           py::arg("scan_duration") = -1.0,
           py::arg("stamp_is_end") = false)
      .def("latest_pose", &VoxelSlam::latest_pose)
      .def("trajectory", &VoxelSlam::trajectory)
      .def("metrics", &VoxelSlam::metrics)
      .def("status", &VoxelSlam::status)
      .def_property_readonly("latest_imu_ticket", &VoxelSlam::latest_imu_ticket)
      .def_property_readonly("latest_prior_ticket", &VoxelSlam::latest_prior_ticket)
      .def_property_readonly("latest_lidar_ticket", &VoxelSlam::latest_lidar_ticket)
      .def("wait_for_processed",
           &VoxelSlam::wait_for_processed,
           py::arg("ticket") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("synchronize",
           &VoxelSlam::synchronize,
           py::arg("ticket") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("pop_deskewed_scans", &VoxelSlam::pop_deskewed_scans)
      .def("request_finish", &VoxelSlam::request_finish)
      .def("is_finished", &VoxelSlam::is_finished)
      .def("finish",
           &VoxelSlam::finish,
           py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::reference_internal);
}

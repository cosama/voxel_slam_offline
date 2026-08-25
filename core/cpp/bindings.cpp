#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  std::uint64_t latest_lidar_ticket = 0;
  std::uint64_t latest_lidar_processed_ticket = 0;
  std::uint64_t lidar_completed = 0;
  std::uint64_t lidar_completed_with_pose = 0;
  std::uint64_t lidar_completed_without_pose = 0;
  bool odometry_waiting_for_imu = false;
  std::uint64_t odometry_waiting_lidar_ticket = 0;
  bool odometry_thread_finished = false;
  bool loop_thread_finished = false;
  bool gba_thread_finished = false;
};

struct Recorder {
  std::mutex mutex;
  std::vector<PoseRecord> poses;
  std::vector<PoseRecord> optimized_poses;
  std::vector<EventRecord> events;
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

void record_odometry_waiting_for_imu(std::uint64_t ticket, bool waiting) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  rec.pipeline.odometry_waiting_for_imu = waiting;
  rec.pipeline.odometry_waiting_lidar_ticket = ticket;
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

  double gba_voxel_size = 2.0;
  double gba_min_eigen_value = 0.01;
  std::vector<double> gba_eigen_value_array = {9.0, 9.0, 9.0, 9.0};
  int gba_total_max_iter = 3;

  bool emit_deskewed_points = false;
  bool enable_loop_closure = true;
  bool enable_global_mapping = true;
};

struct Result {
  std::vector<voxelslam_offline::PoseRecord> poses;
  voxelslam_offline::Metrics metrics;
  std::vector<voxelslam_offline::EventRecord> events;
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
                               std::size_t pending_lidar,
                               std::size_t pending_loop_scanposes,
                               bool loop_processing,
                               bool loop_update_pending,
                               bool loop_reset_pending,
                               bool loop_enabled,
                               bool gba_enabled) {
  py::dict out;

  py::dict imu;
  imu["latest_ticket"] = status.latest_imu_ticket;
  imu["pending_queue"] = pending_imu;
  out["imu"] = imu;

  py::dict lidar;
  lidar["latest_ticket"] = status.latest_lidar_ticket;
  lidar["latest_processed_ticket"] = status.latest_lidar_processed_ticket;
  lidar["completed"] = status.lidar_completed;
  lidar["completed_with_pose"] = status.lidar_completed_with_pose;
  lidar["completed_without_pose"] = status.lidar_completed_without_pose;
  lidar["pending_queue"] = pending_lidar;
  out["lidar"] = lidar;

  py::dict odometry;
  odometry["waiting_for_imu"] = status.odometry_waiting_for_imu;
  odometry["waiting_lidar_ticket"] = status.odometry_waiting_lidar_ticket;
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
      try {
        finish(1.0);
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
      std::fprintf(stderr, "dropping empty lidar sweep at %.9f after filtering\n", stamp);
      return 0;
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
      std::fprintf(stderr, "dropping lidar sweep at %.9f: no points within scan_duration\n", stamp);
      return 0;
    }
    const double begin_stamp = stamp_is_end ? stamp - cloud->back().curvature : stamp;

    const std::uint64_t ticket = ++latest_lidar_ticket_;
    voxelslam_offline::record_lidar_pushed(ticket);
    std::lock_guard<std::mutex> lock(mBuf);
    time_buf.push_back(begin_stamp);
    pcl_buf.push_back(cloud);
    lidar_ticket_buf.push_back(ticket);
    return ticket;
  }

  const Result& finish(double timeout_seconds = 30.0) {
    if (finished_) {
      return result_;
    }

    std::exception_ptr finish_error;
    try {
      wait_for_processed(latest_lidar_ticket_, timeout_seconds);
    } catch (...) {
      finish_error = std::current_exception();
    }
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

    result_.poses = voxelslam_offline::take_poses();
    result_.metrics = voxelslam_offline::snapshot_metrics();
    result_.events = voxelslam_offline::snapshot_events();
    finished_ = true;
    if (finish_error) {
      std::rethrow_exception(finish_error);
    }
    throw_if_thread_error();
    return result_;
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

  std::uint64_t latest_lidar_ticket() const {
    return latest_lidar_ticket_;
  }

  void wait_for_processed(std::uint64_t ticket = 0, double timeout_seconds = -1.0) const {
    throw_if_thread_error();
    const std::uint64_t target = ticket == 0 ? latest_lidar_ticket_ : ticket;
    if (target == 0) {
      return;
    }
    const bool has_timeout = timeout_seconds >= 0.0;
    const auto timeout = std::chrono::duration<double>(std::max(0.0, timeout_seconds));
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      throw_if_thread_error();
      const auto status = voxelslam_offline::snapshot_pipeline_status();
      if (status.latest_lidar_processed_ticket >= target && loop_closure_idle()) {
        return;
      }
      if (has_timeout && std::chrono::steady_clock::now() - start > timeout) {
        throw std::runtime_error("timed out waiting for VoxelSLAM lidar processing and loop closure idle");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
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
      slam_->gba_flag = 0;
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

  bool loop_closure_idle() const {
    if (!options_.enable_loop_closure || !slam_) {
      return true;
    }
    std::lock_guard<std::mutex> lock(slam_->mtx_loop);
    return slam_->buf_lba2loop.empty() &&
           slam_->loop_processing == 0 &&
           slam_->loop_detect == 0 &&
           slam_->reset_flag == 0;
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
    bool loop_processing = false;
    bool loop_update_pending = false;
    bool loop_reset_pending = false;
    if (slam_) {
      std::lock_guard<std::mutex> lock(slam_->mtx_loop);
      pending_loop_scanposes = slam_->buf_lba2loop.size();
      loop_processing = slam_->loop_processing != 0;
      loop_update_pending = slam_->loop_detect != 0;
      loop_reset_pending = slam_->reset_flag != 0;
    }

    return status_to_dict(voxelslam_offline::snapshot_pipeline_status(),
                          pending_imu,
                          pending_lidar,
                          pending_loop_scanposes,
                          loop_processing,
                          loop_update_pending,
                          loop_reset_pending,
                          options_.enable_loop_closure,
                          options_.enable_global_mapping);
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
      .def_readwrite("gba_voxel_size", &VoxelSlamOptions::gba_voxel_size)
      .def_readwrite("gba_min_eigen_value", &VoxelSlamOptions::gba_min_eigen_value)
      .def_readwrite("gba_eigen_value_array", &VoxelSlamOptions::gba_eigen_value_array)
      .def_readwrite("gba_total_max_iter", &VoxelSlamOptions::gba_total_max_iter)
      .def_readwrite("emit_deskewed_points", &VoxelSlamOptions::emit_deskewed_points)
      .def_readwrite("enable_loop_closure", &VoxelSlamOptions::enable_loop_closure)
      .def_readwrite("enable_global_mapping", &VoxelSlamOptions::enable_global_mapping);

  py::class_<Result>(m, "Result")
      .def_property_readonly("trajectory", [](const Result& result) {
        return poses_to_array(result.poses);
      })
      .def_property_readonly("metrics", [](const Result& result) {
        return metrics_to_dict(result.metrics);
      })
      .def_property_readonly("events", [](const Result& result) {
        return events_to_list(result.events);
      });

  py::class_<VoxelSlam>(m, "VoxelSlam")
      .def(py::init<const VoxelSlamOptions&>(), py::arg("options") = VoxelSlamOptions())
      .def("push_imu", &VoxelSlam::push_imu,
           py::arg("stamp"),
           py::arg("linear_acceleration"),
           py::arg("angular_velocity"))
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
      .def_property_readonly("latest_lidar_ticket", &VoxelSlam::latest_lidar_ticket)
      .def("wait_for_processed",
           &VoxelSlam::wait_for_processed,
           py::arg("ticket") = 0,
           py::arg("timeout_seconds") = -1.0,
           py::call_guard<py::gil_scoped_release>())
      .def("pop_deskewed_scans", &VoxelSlam::pop_deskewed_scans)
      .def("request_finish", &VoxelSlam::request_finish)
      .def("is_finished", &VoxelSlam::is_finished)
      .def("finish",
           &VoxelSlam::finish,
           py::arg("timeout_seconds") = 30.0,
           py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::reference_internal);
}

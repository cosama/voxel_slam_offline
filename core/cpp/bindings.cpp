#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <exception>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <pcl/io/pcd_io.h>

#include <voxelslam/offline_bridge.hpp>

#include "BTC.cpp"
#include "voxelslam.cpp"

namespace py = pybind11;

namespace voxelslam_offline {

struct Recorder {
  std::mutex mutex;
  std::vector<PoseRecord> poses;
  std::vector<PoseRecord> optimized_poses;
  std::vector<PointRecord> map_points;
  std::deque<std::vector<PointRecord>> deskewed_scans;
  bool collect_map = false;
  std::size_t max_map_points = 0;
  bool emit_deskewed_points = false;
};

Recorder& recorder() {
  static Recorder instance;
  return instance;
}

void reset_records(bool collect_map,
                   std::size_t max_map_points,
                   bool emit_deskewed_points) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  std::vector<PoseRecord>().swap(rec.poses);
  std::vector<PoseRecord>().swap(rec.optimized_poses);
  std::vector<PointRecord>().swap(rec.map_points);
  rec.deskewed_scans.clear();
  rec.collect_map = collect_map;
  rec.max_map_points = max_map_points;
  rec.emit_deskewed_points = emit_deskewed_points;
}

bool is_emitting_deskewed_points() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return rec.emit_deskewed_points;
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

void record_map_point(double stamp, float x, float y, float z, float intensity) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  if (!rec.collect_map) {
    return;
  }
  if (rec.max_map_points != 0 && rec.map_points.size() >= rec.max_map_points) {
    return;
  }
  rec.map_points.push_back({stamp, x, y, z, intensity});
}

void record_dense_deskewed_points(const std::vector<PointRecord>& points) {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  if (!rec.emit_deskewed_points || points.empty()) {
    return;
  }
  rec.deskewed_scans.push_back(points);
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

std::vector<PointRecord> take_map_points() {
  auto& rec = recorder();
  std::lock_guard<std::mutex> lock(rec.mutex);
  return std::move(rec.map_points);
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

  bool collect_map = false;
  std::size_t max_map_points = 0;
  bool emit_deskewed_points = false;
  bool enable_loop_closure = true;
  bool enable_global_mapping = true;
};

struct Result {
  std::vector<voxelslam_offline::PoseRecord> poses;
  std::vector<voxelslam_offline::PointRecord> map_points;
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

class VoxelSlam {
 public:
  explicit VoxelSlam(const VoxelSlamOptions& options) : options_(options) {
    validate_options(options_);
    std::filesystem::create_directories(options_.save_path);
    reset_upstream_buffers();
    voxelslam_offline::reset_records(options_.collect_map,
                                     options_.max_map_points,
                                     options_.emit_deskewed_points);
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
    try {
      if (!finished_) {
        finish(1.0);
      }
    } catch (...) {
    }
  }

  void push_imu(double stamp,
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

    std::lock_guard<std::mutex> lock(mBuf);
    imu_last_time = stamp;
    imu_buf.push_back(msg);
  }

  void push_lidar(double stamp,
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

    float max_time = 0.0f;
    for (float t : relative_times) {
      if (std::isfinite(t)) {
        max_time = std::max(max_time, t);
      }
    }
    const double begin_stamp = stamp_is_end ? stamp - static_cast<double>(max_time) : stamp;

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
      PointType point;
      point.x = 0.0f;
      point.y = 0.0f;
      point.z = 0.0f;
      point.intensity = 0.0f;
      point.curvature = 0.0f;
      cloud->push_back(point);
      point.curvature = 0.09f;
      cloud->push_back(point);
    }

    std::sort(cloud->begin(), cloud->end(), [](const PointType& a, const PointType& b) {
      return a.curvature < b.curvature;
    });
    while (!cloud->empty() && cloud->back().curvature > 0.11f) {
      cloud->points.pop_back();
    }

    std::lock_guard<std::mutex> lock(mBuf);
    time_buf.push_back(begin_stamp);
    pcl_buf.push_back(cloud);
  }

  const Result& finish(double timeout_seconds = 30.0) {
    if (finished_) {
      return result_;
    }

    wait_for_processing(timeout_seconds);
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
    result_.map_points = voxelslam_offline::take_map_points();
    finished_ = true;
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
  }

  void run_loop() {
    try {
      slam_->thd_loop_closure(node_);
    } catch (...) {
      capture_thread_exception();
    }
  }

  void run_gba() {
    try {
      slam_->thd_globalmapping(node_);
    } catch (...) {
      capture_thread_exception();
    }
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

  std::vector<voxelslam_offline::PoseRecord> current_poses() const {
    if (finished_) {
      return result_.poses;
    }
    return voxelslam_offline::snapshot_best_poses();
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
    imu_last_time = -1.0;
    last_pcl_time = -1.0;
    point_notime = 0;
  }

  void reset_window_permutation() {
    delete[] mp;
    mp = new int[slam_->win_size];
    for (int i = 0; i < slam_->win_size; ++i) {
      mp[i] = i;
    }
  }

  bool pending_input_or_loop_work() const {
    {
      std::lock_guard<std::mutex> lock(mBuf);
      if (!pcl_buf.empty() || !time_buf.empty()) {
        return true;
      }
    }
    {
      std::lock_guard<std::mutex> lock(slam_->mtx_loop);
      if (options_.enable_loop_closure && !slam_->buf_lba2loop.empty()) {
        return true;
      }
    }
    return false;
  }

  void wait_for_processing(double timeout_seconds) const {
    const auto timeout = std::chrono::duration<double>(std::max(0.0, timeout_seconds));
    const auto start = std::chrono::steady_clock::now();
    while (pending_input_or_loop_work()) {
      if (has_thread_error()) {
        break;
      }
      if (std::chrono::steady_clock::now() - start > timeout) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  VoxelSlamOptions options_;
  ros::NodeHandle node_;
  std::unique_ptr<VOXEL_SLAM> slam_;
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
      .def_readwrite("collect_map", &VoxelSlamOptions::collect_map)
      .def_readwrite("max_map_points", &VoxelSlamOptions::max_map_points)
      .def_readwrite("emit_deskewed_points", &VoxelSlamOptions::emit_deskewed_points)
      .def_readwrite("enable_loop_closure", &VoxelSlamOptions::enable_loop_closure)
      .def_readwrite("enable_global_mapping", &VoxelSlamOptions::enable_global_mapping);

  py::class_<Result>(m, "Result")
      .def_property_readonly("trajectory", [](const Result& result) {
        return poses_to_array(result.poses);
      })
      .def_property_readonly("map_points", [](const Result& result) {
        return points_to_array(result.map_points);
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
      .def("pop_deskewed_scans", &VoxelSlam::pop_deskewed_scans)
      .def("request_finish", &VoxelSlam::request_finish)
      .def("is_finished", &VoxelSlam::is_finished)
      .def("finish",
           &VoxelSlam::finish,
           py::arg("timeout_seconds") = 30.0,
           py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::reference_internal);
}

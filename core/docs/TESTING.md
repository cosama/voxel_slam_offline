# Testing Notes

## Build

Built on Ubuntu 24.04 with:

```bash
python3 -m pip install --user --break-system-packages -e ./core
```

Native build dependencies used during this test:

- `libeigen3-dev`
- `libpcl-dev`
- `libgtsam-dev`
- `pybind11-dev`

`libgtsam-dev` on Ubuntu 24.04 has a broken CMake export referencing a missing
`libCppUnitLite.a`, so this package links `libgtsam` directly instead of using
`find_package(GTSAM)`.

## Smoke Tests

- Empty session import/finish succeeds and returns `trajectory.shape == (0, 8)`.
- Synthetic non-overlapping lidar sweeps with IMU produce a trajectory and map
  through the ROS-free Python feed path.
- Overlapping lidar sweep timestamps raise a Python `RuntimeError` instead of
  terminating the process.
- Canonical benchmark subset, odometry-only: 300 lidar sweeps, 6423 IMU samples,
  298 returned poses, 2.682 m path.

## Full Canonical Check

Dataset:
`/var/home/marco/Work/Software/plain_slam_ros2/benchmarks/datasets/nglamp_test_data/canonical/ros2`

Input contract used:

- PointCloud2 header stamp is sweep start time.
- Point field `time` is per-point relative offset in seconds.
- IMU and lidar were fed from Python using `rosbags`; `rosbags` is a test-only
  dependency, not a package runtime dependency.

Result:

- Fed 2706 lidar sweeps and 57455 IMU samples.
- ROS-free output: 2704 poses, 92.037429 m path.
- Original ROS Voxel-SLAM output: 2703 poses, 92.080567 m path.
- Position RMSE against original Voxel-SLAM trajectory, interpolated by
  timestamp: 0.009745 m.
- 95th percentile position difference: 0.021013 m.
- Max position difference: 0.074444 m.

The full-run trajectory was written during testing to
`/tmp/voxelslam_offline_full_trajectory.csv`.

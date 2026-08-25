# Voxel-SLAM Core

Importable `voxelslam` library and native C++ extension.

This package exposes Voxel-SLAM as a ROS-free Python API. It accepts IMU samples,
lidar sweeps, and one explicit IMU/lidar extrinsic transform. It returns pose
snapshots, trajectories, optional deskewed scan batches, status, and final
metrics.

## Install

From the repository root:

```bash
pip install ./core
```

The Python dependency is NumPy. The native extension still depends on the C++
libraries required by upstream Voxel-SLAM, including PCL, Eigen, and GTSAM.

## API Contract

Main public objects:

- `VoxelSlamConfig`: dataclass containing SLAM parameters only.
- `VoxelSlam`: long-lived SLAM instance.
- `Result`: returned by `finish()`, with `trajectory` and `metrics`.
- `DenseMapBuffer`, `PointCloudBuffer`, `BinaryPlyWriter`: optional point-cloud
  output helpers.
- `UrdfTransforms`, `pointcloud_to_numpy`, `write_trajectory_csv`: small I/O
  utilities used by frontends.

Core usage:

```python
slam = voxelslam.VoxelSlam(config, lidar_to_imu=T_imu_lidar)
slam.push_imu(stamp, [ax, ay, az], [gx, gy, gz])
ticket = slam.push_lidar(stamp, points_xyz, relative_times, intensities)
slam.wait_for_processed(ticket)
result = slam.finish()
```

`finish()` is for shutdown. For online-style operation, read `latest_pose()`,
`trajectory()`, `status()`, and `metrics()` while the instance remains active.

## Notes

The algorithm source is pinned as the `upstream/Voxel-SLAM` submodule. CMake
copies it to `build/upstream_staged` and applies the checked-in patch series
from `patches/integration/`; the submodule itself remains pristine. See
`UPSTREAM.md` for the pin, patch inventory, and upgrade procedure.

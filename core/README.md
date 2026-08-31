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
slam.synchronize(ticket)
result = slam.finish()
```

An odometry prior is a causal input, not constructor-owned trajectory data:

```python
slam = voxelslam.VoxelSlam(config, lidar_to_imu=T_imu_lidar, enable_prior=True)
slam.push_prior_pose(stamp, [x, y, z], [qx, qy, qz, qw], covariance_6x6)
```

Push through two samples strictly newer than each sweep end: the first brackets
the pose and the second finalizes that bracket's central-difference velocity.
The worker parks without extrapolating until that lookahead arrives. `finish()` closes the prior
stream and lets uncovered edges fall back to upstream prediction.

`finish()` is for shutdown. For online-style operation, read `latest_pose()`,
`trajectory()`, `status()`, and `metrics()` while the instance remains active.

Deterministic replay uses one producer and one live `VoxelSlam` instance. Push
a sweep, call `synchronize(ticket)`, and call `finish()` when the data runs
out; the caller never pre-checks whether a sweep is coverable. The barrier
blocks until the workers can make no further progress with what has been
submitted -- the ticket completing with every worker drained, or full
quiescence with the estimator parked on a sweep the submitted IMU or prior does
not reach. It takes no timeout and has no bypass, so a replay does not change with
machine speed. Sweeps that were never coverable (a recording that ends without
trailing IMU) appear in `status()["lidar"]["uncovered"]`. Online callers may
omit the call and retain upstream-style asynchronous queue processing. Multiple
live instances and concurrent `push_*()`/`finish()` calls are not supported.

## Notes

The algorithm source is pinned as the `upstream/Voxel-SLAM` submodule. CMake
copies it to `build/upstream_staged` and applies the checked-in patch series
from `patches/integration/`; the submodule itself remains pristine. See
`UPSTREAM.md` for the pin, patch inventory, and upgrade procedure.

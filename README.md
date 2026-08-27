# Voxel-SLAM Offline

ROS-free Python package for running
[HKU-MARS Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM) on decoded lidar
sweeps and IMU samples.

The package keeps the upstream SLAM algorithm close to its original C++ source,
but removes ROS from the public interface. Callers pass plain NumPy/Python data
and receive trajectories, optional deskewed point clouds, status, and aggregate
metrics.

## Layout

- `core/`: importable `voxelslam` library and native C++ extension.
- `runner/`: optional bag frontend using `rosbags`; reads bags and URDF, writes
  trajectory CSV and optional dense PLY.
- `docker/`: container build for the library plus runner.
- `upstream/Voxel-SLAM/`: pinned upstream submodule; the build copies its
  sources into the build tree and applies `core/patches/integration/` there.

## Install

Install the library only:

```bash
pip install ./core
```

Install the optional bag runner as well:

```bash
pip install ./core ./runner
```

The core package has no ROS1 or ROS2 runtime dependency. Native dependencies
still include the libraries used by Voxel-SLAM itself, notably PCL, Eigen, and
GTSAM.

Clone with submodules (or run `git submodule update --init --recursive`) before
building. Upstream provenance, patch policy, and the upgrade procedure are in
`core/UPSTREAM.md`.

## Python API

The library API does one job: run SLAM from already decoded sensor data.
Extrinsics are passed explicitly as a 4x4 transform and are not stored in the
config object.

```python
import numpy as np
import voxelslam

config = voxelslam.VoxelSlamConfig(
    blind=2.8,
    point_filter_num=3,
    enable_loop_closure=True,
    enable_global_mapping=True,
)

# p_imu = T_imu_lidar @ p_lidar
lidar_to_imu = np.array([
    [1.0, 0.0, 0.0, -0.114],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, -0.05],
    [0.0, 0.0, 0.0, 1.0],
])

slam = voxelslam.VoxelSlam(config, lidar_to_imu=lidar_to_imu)
slam.push_imu(stamp, [ax, ay, az], [gx, gy, gz])
ticket = slam.push_lidar(
    stamp=sweep_start_time,
    points=np.asarray(points_xyz, dtype=np.float32),
    relative_times=np.asarray(point_offsets_s, dtype=np.float32),
    intensities=np.asarray(intensities, dtype=np.float32),
)
slam.synchronize(ticket)
result = slam.finish()

trajectory = result.trajectory  # Nx8: stamp,x,y,z,qx,qy,qz,qw
metrics = result.metrics
```

For online-style use, keep one `VoxelSlam` instance alive and do not call
`finish()` until shutdown:

```python
pose = slam.latest_pose()
trajectory = slam.trajectory()
metrics = slam.metrics()
status = slam.status()
deskewed_scans = slam.pop_deskewed_scans()
```

Deterministic replay requires one producer and one live instance: provide IMU
lookahead beyond each sweep, push one sweep, then call `synchronize(ticket)`
before submitting the next. Omitting the barrier preserves asynchronous
upstream-style processing. Concurrent `push_*()`/`finish()` calls and multiple
live instances are not supported.

`push_lidar()` expects points in the lidar frame. Per-point `relative_times`
are seconds from the sweep start. If the input stamp marks the end of the
sweep, pass `stamp_is_end=True`.

## Dense Maps

Set `config.emit_deskewed_points = True` to receive Voxel-SLAM's internally
deskewed scan points. These batches are not retained by the C++ library; drain
them with `pop_deskewed_scans()`.

The helper `DenseMapBuffer` can spool deskewed scans in memory or to a temporary
binary file, then write a final binary PLY after `finish()` returns the optimized
trajectory.

## Bag Runner

The optional runner is intentionally thin dataset plumbing. It reads a ROS1 or
ROS2 bag through `rosbags`, reads fixed-joint URDF extrinsics, feeds the core
library, and writes:

- `trajectory.csv`
- `manifest.json`
- `map.ply` when dense PLY output is enabled

Run:

```bash
voxelslam-run-bag BAG system.urdf config.json output_dir
```

The runner config is a flat JSON object containing any `VoxelSlamConfig` fields
plus runner fields:

```json
{
  "imu_topic": "/imu_ros",
  "points_topic": "/velodyne/velodyne_points",
  "imu_frame": "imu_link",
  "lidar_frame": "velodyne",
  "scan_duration": 0.1,
  "stamp_is_end": false,
  "dense_ply": true,
  "dense_memory_limit_gb": 4.0,
  "point_filter_num": 1
}
```

## Docker

Build the local image:

```bash
docker build -f docker/Dockerfile -t voxel-slam-offline .
```

Run the bag frontend:

```bash
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  voxel-slam-offline \
  voxelslam-run-bag /data/bag /data/system.urdf /data/config.json /output
```

The GitHub Actions workflow publishes:

```bash
ghcr.io/cosama/voxel_slam_offline:<tag>
```

## License

This package follows upstream Voxel-SLAM and is licensed as GPL-2.0-only because
it contains and links against HKU-MARS Voxel-SLAM code. If this creates a
practical issue for your use case, please reach out so we can discuss options.

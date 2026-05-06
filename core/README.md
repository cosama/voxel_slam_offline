# Voxel-SLAM Offline

ROS-free offline Python bindings for the HKU-MARS Voxel-SLAM core.

This package intentionally does one job: run Voxel-SLAM on IMU samples and
decoded lidar sweeps. It does not parse ROS bags, URDF, TF, YAML, or ROS
parameters. Build those inputs outside the package and pass plain Python data.

## Current Scope

- No ROS1 or ROS2 build/runtime dependency.
- PCL, Eigen and GTSAM are still native dependencies.
- Python feeds IMU samples and already decoded lidar sweeps.
- Point clouds must remain in the lidar frame.
- The caller supplies the numeric IMU/lidar extrinsic.
- Lidar timestamps are explicit: pass the sweep stamp plus per-point relative
  times in seconds. If a source header represents the end of a sweep, call
  `push_lidar(..., stamp_is_end=True)`.

The vendored upstream reference is
[hku-mars/Voxel-SLAM](https://github.com/hku-mars/Voxel-SLAM) at commit
`70fc8a28d63823d5989ff184daeea0787b672398`.

## License

This package is licensed under GPL-2.0-only because it contains and links
against upstream HKU-MARS Voxel-SLAM code. If this creates a practical issue
for your use case, please reach out so we can discuss options.

## Minimal Usage

```python
import numpy as np
import voxelslam

config = voxelslam.VoxelSlamConfig()
config.lidar_type = "velodyne"
config.blind = 2.8
config.point_filter_num = 3
config.collect_map = False
config.enable_loop_closure = True
config.enable_global_mapping = True

# p_imu = T_imu_lidar * p_lidar
lidar_to_imu = np.array([
    [1.0, 0.0, 0.0, -0.114],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, -0.05],
    [0.0, 0.0, 0.0, 1.0],
])

slam = voxelslam.VoxelSlam(config, lidar_to_imu=lidar_to_imu)
slam.push_imu(stamp, [ax, ay, az], [gx, gy, gz])
slam.push_lidar(
    stamp=sweep_start_time,
    points=np.asarray(points_xyz, dtype=np.float32),
    relative_times=np.asarray(point_offsets_s, dtype=np.float32),
    intensities=np.asarray(intensity, dtype=np.float32),
)
result = slam.finish()

trajectory = result.trajectory  # Nx8: stamp,x,y,z,qx,qy,qz,qw
map_points = result.map_points  # Nx5: stamp,x,y,z,intensity; empty unless collected
```

For online use, keep one `VoxelSlam` instance alive and read pose snapshots
without calling `finish()`:

```python
pose = slam.latest_pose()    # shape (8,), or None before the first pose
path = slam.trajectory()     # best available Nx8 trajectory
scans = slam.pop_deskewed_scans()  # list of Nx5 stamp,x,y,z,intensity arrays
```

`finish()` is only for shutdown.

Transforms are 4x4 homogeneous matrices. If the surrounding application has an
IMU-to-lidar transform instead, pass it as `imu_to_lidar`; the wrapper inverts
it before calling upstream Voxel-SLAM.

```python
slam = voxelslam.VoxelSlam(config, imu_to_lidar=T_lidar_imu)
```

`imu_to_lidar` means:

```text
p_lidar = R_lidar_imu * p_imu + t_lidar_imu
```

## Config

```python
config = voxelslam.VoxelSlamConfig(
    lidar_type="velodyne",
    point_filter_num=3,
    emit_deskewed_points=False,
)
```

`VoxelSlamConfig` is the public configuration interface. It is a single
standard-library dataclass with sane defaults. Extrinsics are not part of the
config object; pass the transform as the dedicated constructor argument so
calibration has a single explicit path.

For short smoke tests or odometry-only experiments:

```python
config = voxelslam.VoxelSlamConfig(
    enable_loop_closure=False,
    enable_global_mapping=False,
)
```

For per-scan internal-deskew point output:

```python
config = voxelslam.VoxelSlamConfig()
config.point_filter_num = 1
config.emit_deskewed_points = True

for scan in slam.pop_deskewed_scans():
    # scan columns: stamp,x,y,z,intensity in the scan-end lidar frame
    ...
```

The ROS-free API returns trajectory and point data to Python. Writing CSV, PLY,
or any benchmark-specific artifacts belongs in the caller. The retained upstream
`is_save_map` and pose-graph file paths still exist for compatibility, but they
are not the preferred package interface.

Deskewed point batches are optional and non-retained: enable
`config.emit_deskewed_points` and drain them with `pop_deskewed_scans()`.
They are returned in the scan-end lidar frame. Use `trajectory()` to assemble
them; after `finish()` this is the final optimized trajectory when available.

## Bag Runner

The offline bag runner now lives in the top-level `runner/` component. The core
library intentionally stays free of bag, URDF, CSV, and PLY dependencies.

## Notes

The first extraction still uses a small internal compatibility layer to keep the
upstream algorithm close to its original structure. It is not a ROS dependency,
but several upstream files still contain ROS-shaped names internally. The next
cleanup pass should split the large upstream translation unit into a conventional
library and remove the compatibility layer from internal type names.

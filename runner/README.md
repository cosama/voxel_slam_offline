# Voxel-SLAM Runner

Optional bag frontend for the ROS-free `voxelslam` library.

The runner owns dataset plumbing only:

- read ROS1/ROS2 bags with `rosbags`
- parse PointCloud2 and IMU messages without ROS
- read fixed-joint URDF extrinsics through `voxelslam.UrdfTransforms`
- write `trajectory.csv`, `manifest.json`, and optional dense `map.ply`

## Install

From this directory:

```bash
pip install ../core
pip install .
```

## Run

```bash
voxelslam-run-bag BAG system.urdf config.json output_dir
```

The config is a flat JSON object. It may contain `VoxelSlamConfig` fields and
these runner fields:

- `imu_topic`
- `points_topic`
- `imu_frame`
- `lidar_frame`
- `scan_duration`
- `stamp_is_end`
- `dense_ply`
- `dense_memory_limit_gb`

The runner is not required for library use. Applications can call
`voxelslam.VoxelSlam` directly with their own data source.

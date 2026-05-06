# Voxel-SLAM Runner

Small offline runner for the ROS-free `voxelslam` library. It owns dataset
plumbing only: ROS1/ROS2 bag reading through `rosbags`, JSON config loading,
and optional dense PLY export through `voxelslam.DenseMapBuffer`.

Install the core library first, then the runner dependencies:

```bash
pip install ../core
pip install .
```

Run:

```bash
voxelslam-run-bag BAG URDF config.json output_dir
```

The core library remains usable without this runner package.

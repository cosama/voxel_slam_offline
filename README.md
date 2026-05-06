# Voxel-SLAM Offline

ROS-free Voxel-SLAM is organized as two top-level components:

- `core/`: the importable `voxelslam` library and native C++ extension.
- `runner/`: a small offline bag runner with dataset, URDF, CSV, and PLY
  plumbing.

Build/install the library only:

```bash
pip install ./core
```

Build/install the library plus runner:

```bash
pip install ./core ./runner
```

The package follows upstream HKU-MARS Voxel-SLAM and is licensed under
GPL-2.0-only. See `core/LICENSE`.

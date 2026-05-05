# Docker

Build from the package root so `.dockerignore` applies and the image contains
only the ROS-free Voxel-SLAM package:

```bash
docker build -f docker/Dockerfile -t voxel-slam-offline .
```

Smoke test:

```bash
docker run --rm voxel-slam-offline
docker run --rm voxel-slam-offline python -c "import voxelslam; print(voxelslam.VoxelSlamConfig())"
```

Run the bag frontend by mounting data and output paths:

```bash
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  voxel-slam-offline \
  voxelslam-run-bag /data/bag /data/system.urdf /data/config.json /output
```

Publishable image name for GitHub Container Registry:

```bash
ghcr.io/<user-or-org>/voxel-slam-offline:<tag>
```

The GitHub Actions workflow builds pull requests without pushing and publishes
on pushes to `main`, tags matching `v*`, and manual dispatches. For this
repository the published image is:

```bash
ghcr.io/cosama/voxel_slam_offline:<tag>
```

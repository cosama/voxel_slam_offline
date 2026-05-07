# Docker

Container build for the core library plus optional bag runner.

Build from the repository root so `.dockerignore` applies:

```bash
docker build -f docker/Dockerfile -t voxel-slam-offline .
```

Smoke test:

```bash
docker run --rm voxel-slam-offline python -c "import voxelslam; print(voxelslam.VoxelSlamConfig())"
```

Run the bag frontend:

```bash
docker run --rm \
  -v /path/to/data:/data \
  -v /path/to/output:/output \
  voxel-slam-offline \
  voxelslam-run-bag /data/bag /data/system.urdf /data/config.json /output
```

Published image:

```bash
ghcr.io/cosama/voxel_slam_offline:<tag>
```

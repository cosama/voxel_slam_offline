from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .ply import BinaryPlyWriter
from .pointcloud import PointCloudBuffer


class DenseMapBuffer:
    """Temporary deskewed-scan buffer for final dense map export.

    Voxel-SLAM emits deskewed scans before the final optimized trajectory is
    known. This buffer stores those scans either in memory or in a temporary
    binary file, then writes them into the final map frame after `finish()`
    returns the final trajectory.
    """

    def __init__(
        self,
        directory: Path,
        *,
        memory_limit_bytes: int | None = 4 * 1024**3,
    ):
        self._points = PointCloudBuffer(
            directory,
            memory_limit_bytes=memory_limit_bytes,
        )

    @property
    def scans(self) -> int:
        return self._points.chunks

    @property
    def points(self) -> int:
        return self._points.points

    @property
    def storage(self) -> str:
        return self._points.storage

    @property
    def byte_size(self) -> int:
        return self._points.byte_size

    @property
    def path(self) -> Path | None:
        return self._points.path

    def append(self, points: np.ndarray) -> None:
        self._points.append(points)

    def drain_from(self, slam: Any) -> int:
        drained = 0
        for scan in slam.pop_deskewed_scans():
            self.append(scan)
            drained += 1
        return drained

    def write_ply(
        self,
        path: Path,
        trajectory: np.ndarray,
        lidar_to_imu: np.ndarray,
    ) -> None:
        with BinaryPlyWriter(path) as writer:
            for points in self.iter_world_points(trajectory, lidar_to_imu):
                writer.write(points)

    def iter_world_points(
        self,
        trajectory: np.ndarray,
        lidar_to_imu: np.ndarray,
    ):
        trajectory = _prepare_trajectory(trajectory)
        lidar_to_imu = np.asarray(lidar_to_imu, dtype=np.float64)
        if lidar_to_imu.shape != (4, 4):
            raise RuntimeError("lidar_to_imu must be a 4x4 transform")

        for scan in self._points.iter_points():
            yield _transform_scan(scan, trajectory, lidar_to_imu)

    def close(self) -> None:
        self._points.close()

    def remove(self) -> None:
        self._points.remove()


def _transform_scan(
    scan: np.ndarray,
    trajectory: np.ndarray,
    lidar_to_imu: np.ndarray,
) -> np.ndarray:
    scan = np.asarray(scan, dtype=np.float64)
    if scan.size == 0:
        return scan.reshape((0, 5))

    pose = _pose_at(trajectory, float(np.max(scan[:, 0])))
    rot_world_imu = _quat_to_matrix(pose[4:8])
    trans_world_imu = pose[1:4]
    rot_imu_lidar = lidar_to_imu[:3, :3]
    trans_imu_lidar = lidar_to_imu[:3, 3]

    out = scan.copy()
    points_imu = out[:, 1:4] @ rot_imu_lidar.T + trans_imu_lidar
    out[:, 1:4] = points_imu @ rot_world_imu.T + trans_world_imu
    return out


def _prepare_trajectory(trajectory: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise RuntimeError("trajectory expects Nx8 columns: stamp,x,y,z,qx,qy,qz,qw")
    trajectory = trajectory[np.isfinite(trajectory).all(axis=1)]
    if trajectory.size == 0:
        raise RuntimeError("cannot assemble dense PLY without a trajectory")
    order = np.argsort(trajectory[:, 0], kind="stable")
    return trajectory[order]


def _pose_at(trajectory: np.ndarray, stamp: float) -> np.ndarray:
    stamps = trajectory[:, 0]
    if stamp <= stamps[0]:
        return trajectory[0]
    if stamp >= stamps[-1]:
        return trajectory[-1]

    idx = int(np.searchsorted(stamps, stamp, side="left"))
    prev_pose = trajectory[idx - 1]
    next_pose = trajectory[idx]
    span = next_pose[0] - prev_pose[0]
    if span <= 0.0:
        return next_pose
    alpha = float((stamp - prev_pose[0]) / span)
    pose = np.empty(8, dtype=np.float64)
    pose[0] = stamp
    pose[1:4] = (1.0 - alpha) * prev_pose[1:4] + alpha * next_pose[1:4]
    pose[4:8] = _slerp(prev_pose[4:8], next_pose[4:8], alpha)
    return pose


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise RuntimeError("zero-length quaternion in trajectory")
    return q / norm


__all__ = ["DenseMapBuffer"]

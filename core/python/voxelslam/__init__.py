from __future__ import annotations

import time
from typing import Any

from ._core import Result, VoxelSlam as _CoreVoxelSlam
from .config import VoxelSlamConfig
from .dense_map import DenseMapBuffer
from .io import (
    UrdfTransforms,
    pointcloud_to_numpy,
    write_trajectory_csv,
)
from .ply import BinaryPlyWriter
from .pointcloud import PointCloudBuffer


class VoxelSlam:
    """Pythonic wrapper around the upstream Voxel-SLAM core.

    The wrapper does not parse URDF, TF, ROS parameters, or config files. Build
    those externally and pass a VoxelSlamConfig plus one 4x4 extrinsic.
    """

    def __init__(
        self,
        config: VoxelSlamConfig | None = None,
        *,
        lidar_to_imu: Any | None = None,
        imu_to_lidar: Any | None = None,
    ) -> None:
        if config is None:
            config = VoxelSlamConfig()
        if not isinstance(config, VoxelSlamConfig):
            raise TypeError("config must be a VoxelSlamConfig")
        self.config = config
        self.options = config.to_options(
            imu_to_lidar=imu_to_lidar,
            lidar_to_imu=lidar_to_imu,
        )
        self._core = _CoreVoxelSlam(self.options)

    def push_imu(
        self,
        stamp: float,
        linear_acceleration: list[float] | tuple[float, float, float],
        angular_velocity: list[float] | tuple[float, float, float],
    ) -> int:
        """Queue one IMU sample and return its monotonically increasing ticket."""

        return self._core.push_imu(stamp, linear_acceleration, angular_velocity)

    def push_lidar(
        self,
        stamp: float,
        points: Any,
        relative_times: Any | None = None,
        intensities: Any | None = None,
        scan_duration: float = -1.0,
        stamp_is_end: bool = False,
    ) -> int:
        """Queue one lidar sweep and return the ticket used by wait_for_processed()."""

        return self._core.push_lidar(
            stamp,
            points,
            relative_times,
            intensities,
            scan_duration,
            stamp_is_end,
        )

    def finish(self, timeout_seconds: float = 30.0) -> Result:
        return self._core.finish(timeout_seconds)

    def request_finish(self) -> None:
        self._core.request_finish()

    def is_finished(self) -> bool:
        return self._core.is_finished()

    def latest_pose(self) -> Any | None:
        """Return latest pose as [stamp, x, y, z, qx, qy, qz, qw], or None."""

        return self._core.latest_pose()

    def trajectory(self) -> Any:
        """Return the best available trajectory as Nx8 [stamp, x, y, z, qx, qy, qz, qw]."""

        return self._core.trajectory()

    def metrics(self) -> dict[str, Any]:
        """Return aggregate SLAM metrics collected so far."""

        return self._core.metrics()

    def status(self) -> dict[str, Any]:
        """Return queue depths, tickets, and worker lifecycle state."""

        return self._core.status()

    @property
    def latest_imu_ticket(self) -> int:
        return self._core.latest_imu_ticket

    @property
    def latest_lidar_ticket(self) -> int:
        return self._core.latest_lidar_ticket

    def wait_for_processed(
        self,
        ticket: int | None = None,
        timeout_seconds: float = -1.0,
        *,
        allow_waiting_for_imu: bool = False,
    ) -> None:
        target = 0 if ticket is None else ticket
        if not allow_waiting_for_imu:
            self._core.wait_for_processed(target, timeout_seconds)
            return
        if target == 0:
            target = self.latest_lidar_ticket
            if target == 0:
                return

        deadline = None if timeout_seconds < 0.0 else time.monotonic() + timeout_seconds
        while True:
            status = self.status()
            if int(status["lidar"]["latest_processed_ticket"]) >= target:
                return
            odometry = status.get("odometry", {})
            if (
                odometry.get("waiting_for_imu", False)
                and 0 < int(odometry.get("waiting_lidar_ticket", 0)) <= target
            ):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for Voxel-SLAM lidar ticket {target}; "
                    f"status={status}"
                )
            time.sleep(0.001)

    def pop_deskewed_scans(self) -> list[Any]:
        """Return and clear pending deskewed scan batches.

        Each batch has columns [stamp, x, y, z, intensity] in the lidar frame
        at the scan end. Apply the chosen trajectory outside the library.
        """

        return self._core.pop_deskewed_scans()


__all__ = [
    "BinaryPlyWriter",
    "DenseMapBuffer",
    "PointCloudBuffer",
    "Result",
    "UrdfTransforms",
    "VoxelSlam",
    "VoxelSlamConfig",
    "pointcloud_to_numpy",
    "write_trajectory_csv",
]

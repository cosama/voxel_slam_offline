from __future__ import annotations

from typing import Any

from ._core import Result, VoxelSlam as _CoreVoxelSlam
from .config import VoxelSlamConfig


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
    ) -> None:
        self._core.push_imu(stamp, linear_acceleration, angular_velocity)

    def push_lidar(
        self,
        stamp: float,
        points: Any,
        relative_times: Any | None = None,
        intensities: Any | None = None,
        scan_duration: float = -1.0,
        stamp_is_end: bool = False,
    ) -> None:
        self._core.push_lidar(
            stamp,
            points,
            relative_times,
            intensities,
            scan_duration,
            stamp_is_end,
        )

    def finish(self, drain_timeout_seconds: float = 30.0) -> Result:
        return self._core.finish(drain_timeout_seconds)

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

    def pop_deskewed_scans(self) -> list[Any]:
        """Return and clear pending deskewed scan batches.

        Each batch has columns [stamp, x, y, z, intensity] in the lidar frame
        at the scan end. Apply the chosen trajectory outside the library.
        """

        return self._core.pop_deskewed_scans()


__all__ = [
    "Result",
    "VoxelSlam",
    "VoxelSlamConfig",
]

from __future__ import annotations

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
    """ROS-free Voxel-SLAM wrapper."""

    def __init__(
        self,
        config: VoxelSlamConfig | None = None,
        *,
        lidar_to_imu: Any | None = None,
        imu_to_lidar: Any | None = None,
        enable_prior: bool = False,
        prior_ba_sigma_rot: float = 0.0,
        prior_ba_sigma_pos: float = 0.0,
        prior_deskew: bool = False,
        emit_deskewed_points: bool = False,
        enable_loop_closure: bool = True,
        enable_global_mapping: bool = True,
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
        self.options.emit_deskewed_points = bool(emit_deskewed_points)
        self.options.enable_loop_closure = bool(enable_loop_closure)
        self.options.enable_global_mapping = bool(enable_global_mapping)
        self.options.enable_prior = bool(enable_prior)
        for name, (rot, pos) in {
            "prior_ba_sigma": (prior_ba_sigma_rot, prior_ba_sigma_pos),
        }.items():
            if rot < 0.0 or pos < 0.0:
                raise ValueError(f"{name} values must be non-negative")
            if (rot > 0.0) != (pos > 0.0):
                raise ValueError(f"{name}_rot and {name}_pos must be set together")
            if rot > 0.0 and not enable_prior:
                raise ValueError(f"{name} requires enable_prior=True")
        self.options.prior_ba_sigma_rot = float(prior_ba_sigma_rot)
        self.options.prior_ba_sigma_pos = float(prior_ba_sigma_pos)
        if prior_deskew and not enable_prior:
            raise ValueError("prior_deskew requires enable_prior=True")
        self.options.prior_deskew = bool(prior_deskew)
        self._core = _CoreVoxelSlam(self.options)

    def push_imu(
        self,
        stamp: float,
        linear_acceleration: list[float] | tuple[float, float, float],
        angular_velocity: list[float] | tuple[float, float, float],
    ) -> int:
        """Queue one IMU sample and return its monotonically increasing ticket."""

        return self._core.push_imu(stamp, linear_acceleration, angular_velocity)

    def push_prior_pose(
        self,
        stamp: float,
        position: Any,
        orientation: Any,
        covariance: Any | None = None,
    ) -> int:
        """Queue one causal prior pose."""

        return self._core.push_prior_pose(stamp, position, orientation, covariance)

    def close_prior(self) -> None:
        """Close the causal prior."""

        self._core.close_prior()

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

    def finish(self) -> Result:
        """Drain the pipeline, join the workers, and return the run result."""

        return self._core.finish()

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
    def latest_prior_ticket(self) -> int:
        return self._core.latest_prior_ticket

    @property
    def latest_lidar_ticket(self) -> int:
        return self._core.latest_lidar_ticket

    def wait_for_processed(self, ticket: int | None = None) -> None:
        """Wait for deterministic quiescence."""

        self._core.wait_for_processed(0 if ticket is None else ticket)

    def synchronize(self, ticket: int | None = None) -> None:
        """Wait for available progress."""

        self.wait_for_processed(ticket)

    def pop_deskewed_scans(self) -> list[Any]:
        """Return pending deskewed scans."""

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

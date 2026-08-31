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
    """Pythonic wrapper around the upstream Voxel-SLAM core.

    The wrapper does not parse URDF, TF, ROS parameters, or config files. Build
    those externally and pass a VoxelSlamConfig plus one 4x4 extrinsic.

    `config` carries upstream parameters. The keyword arguments carry how this
    process runs the estimator: the extrinsic, whether the causal prior input is
    enabled, and the execution-policy switches `emit_deskewed_points` (open the deskewed
    scan tap that feeds dense maps), `enable_loop_closure` and
    `enable_global_mapping` (suppress upstream's optional threads for
    odometry-only ablations; loop closure requires global mapping).
    """

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
        # Execution policy, not upstream configuration: which optional workers
        # run and which output tap is open. Upstream has no such parameters --
        # it always starts the loop and global-mapping threads and never emits
        # deskewed scans -- so these stay off VoxelSlamConfig, which runners
        # serialize into manifests as the record of the upstream config.
        self.options.emit_deskewed_points = bool(emit_deskewed_points)
        self.options.enable_loop_closure = bool(enable_loop_closure)
        self.options.enable_global_mapping = bool(enable_global_mapping)
        self.options.enable_prior = bool(enable_prior)
        # Local BA uses a separate fixed run-level weight. The odometry IEKF's
        # covariance arrives with push_prior_pose(), beside the pose it describes.
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
        # Deskew against the prior instead of IMU dead reckoning. Sharpens every
        # scan before it is voxelized, and does not touch the pose estimate.
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
        """Push one causal odometry-prior sample and return its input ticket.

        The pose is the IMU body in an arbitrary prior-world frame; orientation
        is scalar-last ``[qx, qy, qz, qw]``. Covariance is optional, but when
        supplied it is a fixed 6x6 ``[rotation, position]`` covariance on every
        sample in the run. Rotation errors are body-frame right perturbations;
        position errors are in the prior world frame. Varying covariance is
        rejected deliberately: the coupling uses fixed weights.
        """

        return self._core.push_prior_pose(stamp, position, orientation, covariance)

    def close_prior(self) -> None:
        """Declare the prior stream complete, without ending the run.

        Sweeps past the prior's last sample then fall back to the estimator's
        own prediction immediately, instead of parking for interpolation
        lookahead that will never arrive. A producer that knows it has no more
        samples must call this: otherwise every remaining sweep parks,
        ``synchronize()`` stops being a barrier for the rest of the run, and
        the LiDAR queue grows until ``finish()`` drains it in one burst.
        Idempotent, and implied by ``finish()``.
        """

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
        """Block until the workers can make no further progress on what was fed.

        That is either `ticket` completing with every worker drained, or full
        quiescence: every worker drained and the estimator parked on a sweep it
        cannot advance because the IMU needed to cover it was never submitted.

        There is no timeout and no bypass. Replay is deterministic because the
        host waits here, so a caller that could stop waiting on a deadline --
        or on a bare "the estimator says it is short of IMU" flag, which can be
        true while other work is still draining -- would be racing the workers
        instead. The quiescent exit is neither: it requires the workers to be
        drained *and* the shortfall to be a property of the data submitted, so
        it holds at every instant once it holds at all and a fast host cannot
        reach it early. The wait ends on success, on quiescence, or on a worker
        exception. Both predicates live in C++ (`workers_idle` and
        `blocked_on_unsubmitted_imu`), which read the worker state under the
        worker mutexes, so there is exactly one of each.
        """

        self._core.wait_for_processed(0 if ticket is None else ticket)

    def synchronize(self, ticket: int | None = None) -> None:
        """Wait until the workers have made all the progress the data allows.

        Push data, call this, and when there is no more data call `finish()`.
        That is the whole protocol: the caller never inspects IMU coverage and
        never pre-checks whether a sweep is coverable. Calling this again on a
        drained pipeline simply returns.
        """

        self.wait_for_processed(ticket)

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

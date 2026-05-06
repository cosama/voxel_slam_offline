from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ._core import VoxelSlamOptions


LIDAR_TYPES = {
    "avia": 0,
    "livox": 0,
    "mid360": 0,
    "velodyne": 1,
    "ouster": 2,
    "hesai": 3,
    "robosense": 4,
    "rslidar": 4,
    "tartan": 5,
    "tartanair": 5,
}


@dataclass(slots=True)
class VoxelSlamConfig:
    """Single Python-side config object.

    Extrinsics are intentionally not stored here. Pass `lidar_to_imu` or
    `imu_to_lidar` to `VoxelSlam` so calibration has one explicit path.
    """

    save_path: str = "/tmp/voxel_slam_offline/"
    bagname: str = "offline"
    lidar_type: str | int = "velodyne"
    blind: float = 2.8
    point_filter_num: int = 3
    is_save_map: bool = False

    odom_cov_gyr: float = 0.01
    odom_cov_acc: float = 1.0
    odom_rdw_gyr: float = 0.0001
    odom_rdw_acc: float = 0.0001
    down_size: float = 0.25
    beam_err: float = 0.01
    dept_err: float = 0.01
    voxel_size: float = 2.0
    min_eigen_value: float = 0.01
    degrade_bound: int = 100
    point_notime: bool = False

    win_size: int = 10
    max_layer: int = 2
    lba_cov_gyr: float = 0.01
    lba_cov_acc: float = 1.0
    lba_rdw_gyr: float = 0.0001
    lba_rdw_acc: float = 0.0001
    min_ba_point: int = 1
    plane_eigen_value_thre: list[float] = field(default_factory=lambda: [4.0, 4.0, 4.0, 4.0])
    imu_coef: float = 0.0001
    thread_num: int = 5

    loop_jud_default: float = 0.45
    loop_icp_eigval: float = 15.0
    loop_ratio_drift: float = 0.01
    loop_curr_halt: int = 10
    loop_prev_halt: int = 10
    loop_acsize: int = 2
    loop_mgsize: int = 2
    loop_is_high_fly: int = 0

    gba_voxel_size: float = 2.0
    gba_min_eigen_value: float = 0.01
    gba_eigen_value_array: list[float] = field(default_factory=lambda: [9.0, 9.0, 9.0, 9.0])
    gba_total_max_iter: int = 3

    collect_map: bool = False
    max_map_points: int = 0
    emit_deskewed_points: bool = False
    max_pending_deskewed_scans: int = 16
    enable_loop_closure: bool = True
    enable_global_mapping: bool = True

    def to_options(
        self,
        *,
        lidar_to_imu: Any | None = None,
        imu_to_lidar: Any | None = None,
    ) -> VoxelSlamOptions:
        options = VoxelSlamOptions()
        _copy_config_to_options(self, options)
        _apply_extrinsic(options, lidar_to_imu=lidar_to_imu, imu_to_lidar=imu_to_lidar)
        return options


def _copy_config_to_options(config: VoxelSlamConfig, options: VoxelSlamOptions) -> None:
    errors: list[str] = []
    for name, value in asdict(config).items():
        _set_option(options, name, value, errors)
    if errors:
        raise ValueError("Invalid Voxel-SLAM config:\n" + "\n".join(f"- {error}" for error in errors))


def _apply_extrinsic(
    options: VoxelSlamOptions,
    *,
    lidar_to_imu: Any | None = None,
    imu_to_lidar: Any | None = None,
) -> None:
    if (imu_to_lidar is None) == (lidar_to_imu is None):
        raise ValueError("Provide exactly one of imu_to_lidar or lidar_to_imu")

    if lidar_to_imu is not None:
        transform = _parse_transform(lidar_to_imu, "lidar_to_imu")
    else:
        transform = np.linalg.inv(_parse_transform(imu_to_lidar, "imu_to_lidar"))

    options.extrinsic_rota = transform[:3, :3].reshape(-1).tolist()
    options.extrinsic_tran = transform[:3, 3].tolist()


def _parse_transform(transform: Any, label: str) -> np.ndarray:
    try:
        arr = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric 4x4 transform") from exc
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 transform, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{label} last row must be [0, 0, 0, 1]")
    return arr


def _set_option(options: VoxelSlamOptions, attr: str, value: Any, errors: list[str]) -> None:
    try:
        if attr == "lidar_type" and isinstance(value, str):
            value = _lidar_type_from_string(value)
        current = getattr(options, attr)
        setattr(options, attr, _coerce_value(value, current, attr))
    except Exception as exc:
        errors.append(f"{attr}: {exc}")


def _coerce_value(value: Any, current: Any, attr: str) -> Any:
    if isinstance(current, bool):
        return _coerce_bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool):
            raise TypeError("expected int, got bool")
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    if isinstance(current, list):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError("expected a list")
        if attr in {"plane_eigen_value_thre", "gba_eigen_value_array"}:
            return [float(item) for item in value]
        return list(value)
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise TypeError("integer bool values must be 0 or 1")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"expected bool, got {value!r}")


def _lidar_type_from_string(value: str) -> int:
    key = value.strip().lower().replace("-", "").replace("_", "")
    if key not in LIDAR_TYPES:
        allowed = ", ".join(sorted(LIDAR_TYPES))
        raise ValueError(f"unknown lidar_type {value!r}; allowed: {allowed}")
    return LIDAR_TYPES[key]

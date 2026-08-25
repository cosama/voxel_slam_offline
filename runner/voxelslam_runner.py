from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from voxelslam import (
    DenseMapBuffer,
    UrdfTransforms,
    VoxelSlam,
    VoxelSlamConfig,
    pointcloud_to_numpy,
    write_trajectory_csv,
)


PROGRESS_INTERVAL = 100
IMU_TIME_EPSILON_SECONDS = 1e-6
REQUIRED_TRAILING_IMU_SECONDS = 0.005
PROCESSING_TIMEOUT_SECONDS = 30.0
TRAJECTORY_FRAME_ID = "map"
TRAJECTORY_CHILD_FRAME_ID = "base_link"


@dataclass(slots=True)
class RunnerConfig:
    imu_topic: str | None = None
    points_topic: str | None = None
    imu_frame: str | None = None
    lidar_frame: str | None = None
    scan_duration: float = 0.1
    stamp_is_end: bool = False
    dense_ply: bool = False
    dense_memory_limit_gb: float = 4.0


@dataclass(slots=True)
class BagInfo:
    imu_topic: str
    points_topic: str
    imu_frame: str
    lidar_frame: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ROS-free Voxel-SLAM on a ROS1/ROS2 bag.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("config", type=Path, help="Flat JSON config.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    slam_config, runner_config = load_config(args.config)
    summary = run_bag(args.bag, args.urdf, slam_config, runner_config, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


def load_config(path: Path) -> tuple[VoxelSlamConfig, RunnerConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config JSON must contain an object")

    slam_fields = {field.name for field in fields(VoxelSlamConfig)}
    runner_fields = {field.name for field in fields(RunnerConfig)}
    unknown = sorted(set(payload) - slam_fields - runner_fields)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")

    slam_config = VoxelSlamConfig(**{key: payload[key] for key in payload if key in slam_fields})
    runner_config = RunnerConfig(
        **{key: payload[key] for key in payload if key in runner_fields}
    )
    slam_config.emit_deskewed_points = runner_config.dense_ply
    return slam_config, runner_config


def run_bag(
    bag: Path,
    urdf: Path,
    slam_config: VoxelSlamConfig,
    runner_config: RunnerConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    info = inspect_bag(bag, runner_config)
    lidar_to_imu = UrdfTransforms.read(urdf).get_transform(info.imu_frame, info.lidar_frame)
    slam = VoxelSlam(
        slam_config,
        lidar_to_imu=lidar_to_imu,
    )

    pointcloud_ply = output_dir / "map.ply" if runner_config.dense_ply else None
    dense_map = (
        DenseMapBuffer(
            output_dir,
            memory_limit_bytes=int(runner_config.dense_memory_limit_gb * 1024**3),
        )
        if runner_config.dense_ply
        else None
    )
    imu_count = 0
    lidar_count = 0
    last_imu_stamp = None
    pending_lidar = []
    start_time = time.monotonic()

    try:
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([bag], default_typestore=typestore) as reader:
            imu_conn = get_connection(reader, info.imu_topic, "sensor_msgs/msg/Imu")
            points_conn = get_connection(reader, info.points_topic, "sensor_msgs/msg/PointCloud2")
            for conn, _timestamp_ns, raw in reader.messages(connections=[imu_conn, points_conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == info.imu_topic:
                    stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                    slam.push_imu(
                        stamp,
                        [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                        [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                    )
                    last_imu_stamp = stamp
                    imu_count += 1
                    lidar_count += push_ready_lidar(
                        slam,
                        pending_lidar,
                        dense_map,
                        last_imu_stamp=last_imu_stamp,
                        stamp_is_end=runner_config.stamp_is_end,
                        scan_duration=runner_config.scan_duration,
                    )
                else:
                    stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                    points, times, intensities = parse_pointcloud(
                        msg,
                        stamp,
                        scan_duration=runner_config.scan_duration,
                    )
                    pending_lidar.append((stamp, points, times, intensities))
                    lidar_count += push_ready_lidar(
                        slam,
                        pending_lidar,
                        dense_map,
                        last_imu_stamp=last_imu_stamp,
                        stamp_is_end=runner_config.stamp_is_end,
                        scan_duration=runner_config.scan_duration,
                    )
                    report_progress(dense_map, imu_count, lidar_count, start_time)

        lidar_count += push_ready_lidar(
            slam,
            pending_lidar,
            dense_map,
            last_imu_stamp=last_imu_stamp,
            stamp_is_end=runner_config.stamp_is_end,
            scan_duration=runner_config.scan_duration,
            require_all=True,
        )
        result = slam.finish()
        pipeline_status = slam.status()
        if dense_map is not None:
            dense_map.drain_from(slam)

        trajectory_csv = output_dir / "trajectory.csv"
        write_trajectory_csv(
            trajectory_csv,
            result.trajectory,
            frame_id=TRAJECTORY_FRAME_ID,
            child_frame_id=TRAJECTORY_CHILD_FRAME_ID,
        )
        if pointcloud_ply is not None and dense_map is not None:
            dense_map.write_ply(pointcloud_ply, result.trajectory, lidar_to_imu)
            dense_map.remove()
    finally:
        if dense_map is not None:
            dense_map.close()

    summary = {
        "bag": str(bag),
        "urdf": str(urdf),
        "output_dir": str(output_dir),
        "imu_topic": info.imu_topic,
        "points_topic": info.points_topic,
        "imu_frame": info.imu_frame,
        "lidar_frame": info.lidar_frame,
        "imu_messages": imu_count,
        "lidar_messages": lidar_count,
        "poses": int(result.trajectory.shape[0]),
        "trajectory_csv": str(trajectory_csv),
        "pointcloud_ply": str(pointcloud_ply) if pointcloud_ply else None,
        "dense_scans": dense_map.scans if dense_map is not None else 0,
        "dense_points": dense_map.points if dense_map is not None else 0,
        "metrics": result.metrics,
        "pipeline": pipeline_status,
    }
    manifest = output_dir / "manifest.json"
    summary["manifest"] = str(manifest)
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def push_ready_lidar(
    slam: VoxelSlam,
    pending_lidar: list,
    dense_map: "DenseMapBuffer | None",
    *,
    last_imu_stamp: float | None,
    stamp_is_end: bool,
    scan_duration: float,
    require_all: bool = False,
) -> int:
    pushed = 0
    while pending_lidar:
        stamp, points, times, intensities = pending_lidar[0]
        if points.size == 0:
            pending_lidar.pop(0)
            continue
        scan_end = stamp if stamp_is_end else stamp + scan_duration * 1.1
        if (
            last_imu_stamp is None
            or last_imu_stamp
            <= scan_end + REQUIRED_TRAILING_IMU_SECONDS + IMU_TIME_EPSILON_SECONDS
        ):
            if require_all:
                pending_lidar.pop(0)
                print(
                    "dropping final lidar message without required trailing IMU: "
                    f"last_imu={last_imu_stamp}, scan_end={scan_end}, "
                    f"required={REQUIRED_TRAILING_IMU_SECONDS}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            break
        pending_lidar.pop(0)
        ticket = slam.push_lidar(
            stamp,
            points,
            times,
            intensities,
            scan_duration=scan_duration,
            stamp_is_end=stamp_is_end,
        )
        slam.wait_for_processed(
            ticket,
            PROCESSING_TIMEOUT_SECONDS,
            allow_waiting_for_imu=not require_all,
        )
        if dense_map is not None:
            dense_map.drain_from(slam)
        pushed += 1
    return pushed


def report_progress(
    dense_map: "DenseMapBuffer | None",
    imu_count: int,
    lidar_count: int,
    start_time: float,
) -> None:
    if lidar_count <= 0 or PROGRESS_INTERVAL <= 0 or lidar_count % PROGRESS_INTERVAL != 0:
        return
    elapsed = time.monotonic() - start_time
    dense = ""
    if dense_map is not None:
        dense = (
            f", dense_scans={dense_map.scans}, dense_points={dense_map.points}, "
            f"buffer={dense_map.storage}:{dense_map.byte_size}"
        )
    print(
        f"progress lidar={lidar_count}, imu={imu_count}, elapsed={elapsed:.1f}s{dense}",
        file=sys.stderr,
        flush=True,
    )


def inspect_bag(bag: Path, config: RunnerConfig) -> BagInfo:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag], default_typestore=typestore) as reader:
        imu_conn = select_connection(reader, config.imu_topic, "sensor_msgs/msg/Imu")
        points_conn = select_connection(reader, config.points_topic, "sensor_msgs/msg/PointCloud2")
        imu_frame = clean_frame(config.imu_frame)
        lidar_frame = clean_frame(config.lidar_frame)
        for conn, _timestamp_ns, raw in reader.messages(connections=[imu_conn, points_conn]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == imu_conn.topic and not imu_frame:
                imu_frame = clean_frame(msg.header.frame_id)
            if conn.topic == points_conn.topic and not lidar_frame:
                lidar_frame = clean_frame(msg.header.frame_id)
            if imu_frame and lidar_frame:
                break

    if not imu_frame:
        raise RuntimeError("could not infer IMU frame; set 'imu_frame' in the config")
    if not lidar_frame:
        raise RuntimeError("could not infer lidar frame; set 'lidar_frame' in the config")
    return BagInfo(imu_conn.topic, points_conn.topic, imu_frame, lidar_frame)


def select_connection(reader: AnyReader, topic: str | None, msgtype: str):
    if topic:
        return get_connection(reader, topic, msgtype)
    matches = [conn for conn in reader.connections if conn.msgtype == msgtype]
    if len(matches) != 1:
        topics = ", ".join(conn.topic for conn in matches) or "none"
        raise RuntimeError(f"expected one {msgtype} topic, found {len(matches)}: {topics}")
    return matches[0]


def get_connection(reader: AnyReader, topic: str, msgtype: str):
    matches = [conn for conn in reader.connections if conn.topic == topic and conn.msgtype == msgtype]
    if not matches:
        raise RuntimeError(f"bag does not contain {msgtype} topic {topic!r}")
    return matches[0]


def parse_pointcloud(
    msg,
    stamp: float,
    *,
    scan_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud = pointcloud_to_numpy(msg)
    names = set(cloud.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise RuntimeError("PointCloud2 message does not contain x/y/z fields")

    points = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype("<f4", copy=False)
    intensities = (
        np.asarray(cloud["intensity"], dtype="<f4")
        if "intensity" in names
        else np.zeros(points.shape[0], dtype="<f4")
    )
    times = point_times(cloud, names, stamp, scan_duration).astype("<f4", copy=False)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(intensities) & np.isfinite(times)
    return points[mask], times[mask], intensities[mask]


def point_times(
    cloud: np.ndarray,
    names: set[str],
    stamp: float,
    scan_duration: float,
) -> np.ndarray:
    field = next((name for name in ("time", "timestamp", "t") if name in names), None)
    if field is None:
        return np.linspace(0.0, scan_duration, cloud.shape[0], dtype=np.float64)

    raw = np.asarray(cloud[field], dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros(raw.shape[0], dtype=np.float64)

    max_abs = float(np.max(np.abs(finite)))
    if max_abs > 1e12:
        return raw * 1e-9 - stamp
    if max_abs > stamp - 1000.0:
        return raw - stamp
    if field in {"timestamp", "t"} or max_abs > 1e6:
        return raw * 1e-9
    return raw


def clean_frame(frame: Any | None) -> str:
    if frame is None:
        return ""
    return str(frame).strip().lstrip("/")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

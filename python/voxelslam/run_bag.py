from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from . import VoxelSlam, VoxelSlamConfig


POINTFIELD_DTYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    8: "<f8",
}


@dataclass(slots=True)
class RunnerConfig:
    imu_topic: str | None = None
    points_topic: str | None = None
    imu_frame: str | None = None
    lidar_frame: str | None = None
    scan_duration: float = 0.1
    stamp_is_end: bool = False
    ply: str = "none"
    max_lidar_messages: int = 0
    progress_interval: int = 100
    finish_poll_seconds: float = 0.05
    finish_timeout_seconds: float = 30.0


@dataclass(slots=True)
class Joint:
    parent: str
    translation: np.ndarray
    rotation: np.ndarray


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
    runner_config = RunnerConfig(**{key: payload[key] for key in payload if key in runner_fields})
    runner_config.ply = runner_config.ply.lower()
    if runner_config.ply not in {"none", "map", "dense"}:
        raise ValueError("config key 'ply' must be one of: none, map, dense")

    if runner_config.ply == "map":
        slam_config.collect_map = True
    elif runner_config.ply == "dense":
        slam_config.emit_deskewed_points = True
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
    lidar_to_imu = load_lidar_to_imu(urdf, info.imu_frame, info.lidar_frame)
    slam = VoxelSlam(
        slam_config,
        lidar_to_imu=lidar_to_imu,
    )

    map_ply = output_dir / "map.ply" if runner_config.ply != "none" else None
    dense_spool = DeskewedScanSpool(output_dir) if runner_config.ply == "dense" else None
    imu_count = 0
    lidar_count = 0
    start_time = time.monotonic()

    try:
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([bag], default_typestore=typestore) as reader:
            imu_conn = get_connection(reader, info.imu_topic, "sensor_msgs/msg/Imu")
            points_conn = get_connection(reader, info.points_topic, "sensor_msgs/msg/PointCloud2")
            for conn, timestamp_ns, raw in reader.messages(connections=[imu_conn, points_conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == info.imu_topic:
                    stamp = stamp_to_seconds(msg.header.stamp, timestamp_ns)
                    slam.push_imu(
                        stamp,
                        [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                        [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                    )
                    imu_count += 1
                else:
                    stamp = stamp_to_seconds(msg.header.stamp, timestamp_ns)
                    points, times, intensities = parse_pointcloud(
                        msg,
                        stamp,
                        lidar_type=slam_config.lidar_type,
                        scan_duration=runner_config.scan_duration,
                    )
                    slam.push_lidar(
                        stamp,
                        points,
                        times,
                        intensities,
                        stamp_is_end=runner_config.stamp_is_end,
                    )
                    lidar_count += 1
                    drain_deskewed(slam, dense_spool)
                    report_progress(runner_config, dense_spool, imu_count, lidar_count, start_time)
                    if runner_config.max_lidar_messages and lidar_count >= runner_config.max_lidar_messages:
                        break

        result = finish_and_drain(slam, dense_spool, runner_config, start_time)
        drain_deskewed(slam, dense_spool)

        trajectory_csv = output_dir / "trajectory.csv"
        write_trajectory_csv(trajectory_csv, result.trajectory)
        if runner_config.ply == "map" and map_ply is not None:
            write_ply(map_ply, result.map_points)
        if runner_config.ply == "dense" and map_ply is not None and dense_spool is not None:
            write_dense_ply(map_ply, dense_spool, result.trajectory, lidar_to_imu)
            dense_spool.remove()
    finally:
        if dense_spool is not None:
            dense_spool.close()

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
        "map_ply": str(map_ply) if map_ply else None,
        "dense_scans": dense_spool.scans if dense_spool is not None else 0,
        "dense_points": dense_spool.points if dense_spool is not None else 0,
    }
    manifest = output_dir / "manifest.json"
    summary["manifest"] = str(manifest)
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def drain_deskewed(slam: VoxelSlam, sink: "DeskewedScanSpool | None") -> None:
    if sink is None:
        return
    for scan in slam.pop_deskewed_scans():
        sink.write(scan)


def finish_and_drain(
    slam: VoxelSlam,
    sink: "DeskewedScanSpool | None",
    config: RunnerConfig,
    start_time: float,
):
    if sink is None:
        return slam.finish(config.finish_timeout_seconds)

    result_holder = {}
    error_holder = {}

    def finish_worker() -> None:
        try:
            result_holder["result"] = slam.finish(config.finish_timeout_seconds)
        except BaseException as exc:
            error_holder["error"] = exc

    worker = threading.Thread(target=finish_worker, daemon=True)
    worker.start()
    last_report = time.monotonic()
    while worker.is_alive():
        drain_deskewed(slam, sink)
        now = time.monotonic()
        if config.progress_interval > 0 and now - last_report >= 5.0:
            report_finish_progress(sink, start_time)
            last_report = now
        worker.join(timeout=max(0.001, config.finish_poll_seconds))
    drain_deskewed(slam, sink)
    if error_holder:
        raise error_holder["error"]
    return result_holder["result"]


def report_progress(
    config: RunnerConfig,
    spool: "DeskewedScanSpool | None",
    imu_count: int,
    lidar_count: int,
    start_time: float,
) -> None:
    if config.progress_interval <= 0 or lidar_count % config.progress_interval != 0:
        return
    elapsed = time.monotonic() - start_time
    dense = ""
    if spool is not None:
        dense = f", dense_scans={spool.scans}, dense_points={spool.points}, spool={spool.path.stat().st_size}"
    print(f"progress lidar={lidar_count}, imu={imu_count}, elapsed={elapsed:.1f}s{dense}", file=sys.stderr, flush=True)


def report_finish_progress(spool: "DeskewedScanSpool", start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    print(
        f"finish elapsed={elapsed:.1f}s, dense_scans={spool.scans}, "
        f"dense_points={spool.points}, spool={spool.path.stat().st_size}",
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
        for conn, timestamp_ns, raw in reader.messages(connections=[imu_conn, points_conn]):
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


def load_lidar_to_imu(urdf: Path, imu_frame: str, lidar_frame: str) -> np.ndarray:
    joints, links = load_fixed_joints(urdf)
    rotation, translation = compute_transform_between_links(joints, links, imu_frame, lidar_frame)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def load_fixed_joints(path: Path) -> tuple[dict[str, Joint], set[str]]:
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints: dict[str, Joint] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        origin_el = joint.find("origin")
        if origin_el is not None:
            if "xyz" in origin_el.attrib:
                xyz = np.fromstring(origin_el.attrib["xyz"], sep=" ", dtype=np.float64)
            if "rpy" in origin_el.attrib:
                rpy = np.fromstring(origin_el.attrib["rpy"], sep=" ", dtype=np.float64)
        parent = parent_el.attrib["link"]
        child = child_el.attrib["link"]
        joints[child] = Joint(parent=parent, translation=xyz, rotation=rpy_to_matrix(rpy))
        links.update({parent, child})
    return joints, links


def compute_transform_between_links(
    joints: dict[str, Joint],
    links: set[str],
    parent_link: str,
    child_link: str,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rot, parent_trans, parent_root = transform_from_root(joints, links, parent_link)
    child_rot, child_trans, child_root = transform_from_root(joints, links, child_link)
    if parent_root != child_root:
        raise ValueError(f"URDF links are not connected: {parent_link} and {child_link}")
    return parent_rot.T @ child_rot, parent_rot.T @ (child_trans - parent_trans)


def transform_from_root(
    joints: dict[str, Joint],
    links: set[str],
    link_name: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if link_name not in links:
        raise ValueError(f"link not found in URDF: {link_name}")

    chain: list[Joint] = []
    current = link_name
    visited = {current}
    while current in joints:
        joint = joints[current]
        chain.append(joint)
        current = joint.parent
        if current in visited:
            raise ValueError("detected a loop in the URDF tree")
        visited.add(current)

    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for joint in reversed(chain):
        translation = translation + rotation @ joint.translation
        rotation = rotation @ joint.rotation
    return rotation, translation, current


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def pointcloud_to_numpy(msg) -> np.ndarray:
    names = []
    formats = []
    offsets = []
    byteorder = ">" if msg.is_bigendian else "<"
    for field in msg.fields:
        if field.datatype not in POINTFIELD_DTYPES:
            continue
        names.append(field.name)
        base = POINTFIELD_DTYPES[field.datatype]
        fmt = np.dtype(byteorder + base[1:]) if base.startswith(("<", ">")) else np.dtype(base)
        if field.count != 1:
            fmt = np.dtype((fmt, field.count))
        formats.append(fmt)
        offsets.append(field.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    return np.frombuffer(memoryview(msg.data), dtype=dtype, count=msg.width * msg.height)


def parse_pointcloud(
    msg,
    stamp: float,
    *,
    lidar_type: str | int,
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
    times = point_times(cloud, names, stamp, lidar_type, scan_duration).astype("<f4", copy=False)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(intensities) & np.isfinite(times)
    return points[mask], times[mask], intensities[mask]


def point_times(
    cloud: np.ndarray,
    names: set[str],
    stamp: float,
    lidar_type: str | int,
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
    lidar_name = str(lidar_type).lower()
    if max_abs > 1e12:
        return raw * 1e-9 - stamp
    if max_abs > stamp - 1000.0:
        return raw - stamp
    if field in {"timestamp", "t"} or lidar_name == "livox" or max_abs > 1e6:
        return raw * 1e-9
    return raw


def stamp_to_seconds(stamp, fallback_ns: int) -> float:
    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    if sec == 0 and nanosec == 0:
        return fallback_ns * 1e-9
    return float(sec) + float(nanosec) * 1e-9


def clean_frame(frame: Any | None) -> str:
    if frame is None:
        return ""
    return str(frame).strip().lstrip("/")


def write_trajectory_csv(path: Path, trajectory: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for row in np.asarray(trajectory, dtype=np.float64):
            writer.writerow([f"{value:.9f}" for value in row])


class DeskewedScanSpool:
    """Disk-backed scan queue; keeps dense output out of library memory."""

    def __init__(self, directory: Path):
        stream = tempfile.NamedTemporaryFile(prefix="deskewed_", suffix=".bin", dir=directory, delete=False)
        self.path = Path(stream.name)
        self.scans = 0
        self.points = 0
        self._stream = stream

    def write(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype="<f8")
        if points.ndim != 2 or points.shape[1] != 5:
            raise RuntimeError("deskewed scan expects Nx5 columns: timestamp,x,y,z,intensity")
        np.asarray([points.shape[0]], dtype="<u8").tofile(self._stream)
        points.tofile(self._stream)
        self.scans += 1
        self.points += int(points.shape[0])

    def iter_scans(self):
        self._stream.flush()
        self._stream.seek(0)
        while True:
            count = np.fromfile(self._stream, dtype="<u8", count=1)
            if count.size == 0:
                break
            points = np.fromfile(self._stream, dtype="<f8", count=int(count[0]) * 5)
            if points.size != int(count[0]) * 5:
                raise RuntimeError("truncated deskewed scan spool")
            yield points.reshape((-1, 5))

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def remove(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


POINT_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
    ]
)


def make_vertices(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 5:
        raise RuntimeError("point output expects Nx5 columns: timestamp,x,y,z,intensity")
    points = points[np.isfinite(points).all(axis=1)]
    vertices = np.empty(points.shape[0], dtype=POINT_DTYPE)
    vertices["timestamp"] = points[:, 0]
    vertices["x"] = points[:, 1]
    vertices["y"] = points[:, 2]
    vertices["z"] = points[:, 3]
    vertices["intensity"] = points[:, 4]
    return vertices


def write_ply(path: Path, points: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    PlyData([PlyElement.describe(make_vertices(points), "vertex")], text=False).write(path)


def write_dense_ply(
    path: Path,
    spool: DeskewedScanSpool,
    trajectory: np.ndarray,
    lidar_to_imu: np.ndarray,
) -> None:
    trajectory = prepare_trajectory(trajectory)
    lidar_to_imu = np.asarray(lidar_to_imu, dtype=np.float64)
    if lidar_to_imu.shape != (4, 4):
        raise RuntimeError("lidar_to_imu must be a 4x4 transform")

    writer = StreamingPlyWriter(path)
    try:
        for scan in spool.iter_scans():
            writer.write(transform_deskewed_scan(scan, trajectory, lidar_to_imu))
    finally:
        writer.close()


def transform_deskewed_scan(
    scan: np.ndarray,
    trajectory: np.ndarray,
    lidar_to_imu: np.ndarray,
) -> np.ndarray:
    scan = np.asarray(scan, dtype=np.float64)
    if scan.size == 0:
        return scan.reshape((0, 5))

    pose = pose_at(trajectory, float(np.max(scan[:, 0])))
    rot_world_imu = quat_to_matrix(pose[4:8])
    trans_world_imu = pose[1:4]
    rot_imu_lidar = lidar_to_imu[:3, :3]
    trans_imu_lidar = lidar_to_imu[:3, 3]

    out = scan.copy()
    points_imu = out[:, 1:4] @ rot_imu_lidar.T + trans_imu_lidar
    out[:, 1:4] = points_imu @ rot_world_imu.T + trans_world_imu
    return out


def prepare_trajectory(trajectory: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise RuntimeError("trajectory expects Nx8 columns: stamp,x,y,z,qx,qy,qz,qw")
    trajectory = trajectory[np.isfinite(trajectory).all(axis=1)]
    if trajectory.size == 0:
        raise RuntimeError("cannot assemble dense PLY without a trajectory")
    order = np.argsort(trajectory[:, 0], kind="stable")
    return trajectory[order]


def pose_at(trajectory: np.ndarray, stamp: float) -> np.ndarray:
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
    pose[4:8] = slerp(prev_pose[4:8], next_pose[4:8], alpha)
    return pose


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise RuntimeError("zero-length quaternion in trajectory")
    return q / norm


class StreamingPlyWriter:
    _count_width = 20

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self._stream = path.open("wb+")
        count_token = "0" * self._count_width
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {count_token}\n"
            "property double timestamp\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float intensity\n"
            "end_header\n"
        )
        self._count_offset = header.index(count_token)
        self._stream.write(header.encode("ascii"))

    def write(self, points: np.ndarray) -> None:
        vertices = make_vertices(points)
        self._stream.write(vertices.tobytes(order="C"))
        self.count += int(vertices.shape[0])

    def close(self) -> None:
        if self._stream.closed:
            return
        self._stream.seek(self._count_offset)
        self._stream.write(f"{self.count:0{self._count_width}d}".encode("ascii"))
        self._stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

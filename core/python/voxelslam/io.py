from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_POINTFIELD_DTYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    8: "<f8",
}


@dataclass(frozen=True, slots=True)
class _Joint:
    parent: str
    transform: np.ndarray


class UrdfTransforms:
    """Minimal fixed-joint URDF transform reader.

    `get_transform(target, source)` returns T_target_source, so:

        p_target = T_target_source @ p_source
    """

    def __init__(self, links: set[str], joints: dict[str, _Joint]):
        self.links = frozenset(links)
        self._joints = dict(joints)

    @classmethod
    def read(cls, path: str | Path) -> "UrdfTransforms":
        root = ET.parse(path).getroot()
        links = {_frame_name(link.attrib["name"]) for link in root.findall("link")}
        joints: dict[str, _Joint] = {}

        for joint in root.findall("joint"):
            if joint.attrib.get("type") != "fixed":
                continue
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is None or child_el is None:
                continue

            parent = _frame_name(parent_el.attrib["link"])
            child = _frame_name(child_el.attrib["link"])
            joints[child] = _Joint(parent, _joint_transform(joint.find("origin")))
            links.update({parent, child})

        return cls(links, joints)

    def get_transform(self, target_frame: str, source_frame: str) -> np.ndarray:
        target = _frame_name(target_frame)
        source = _frame_name(source_frame)
        target_transform, target_root = self._transform_from_root(target)
        source_transform, source_root = self._transform_from_root(source)
        if target_root != source_root:
            raise ValueError(f"URDF links are not connected: {target} and {source}")
        return np.linalg.inv(target_transform) @ source_transform

    def _transform_from_root(self, link: str) -> tuple[np.ndarray, str]:
        if link not in self.links:
            raise ValueError(f"link not found in URDF: {link}")

        chain: list[_Joint] = []
        current = link
        visited = {current}
        while current in self._joints:
            joint = self._joints[current]
            chain.append(joint)
            current = joint.parent
            if current in visited:
                raise ValueError("detected a loop in the URDF tree")
            visited.add(current)

        transform = np.eye(4, dtype=np.float64)
        for joint in reversed(chain):
            transform = transform @ joint.transform
        return transform, current


def pointcloud_to_numpy(msg: Any) -> np.ndarray:
    """Parse a ROS1/ROS2 PointCloud2-like message into a structured array."""

    names = []
    formats = []
    offsets = []
    byteorder = ">" if msg.is_bigendian else "<"
    for field in msg.fields:
        if field.datatype not in _POINTFIELD_DTYPES:
            continue
        names.append(field.name)
        base = _POINTFIELD_DTYPES[field.datatype]
        dtype = np.dtype(byteorder + base[1:]) if base.startswith(("<", ">")) else np.dtype(base)
        if field.count != 1:
            dtype = np.dtype((dtype, field.count))
        formats.append(dtype)
        offsets.append(field.offset)

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(memoryview(msg.data), dtype=dtype, count=msg.width * msg.height)


def write_trajectory_csv(
    path: str | Path,
    trajectory: np.ndarray,
    *,
    frame_id: str | None = None,
    child_frame_id: str | None = None,
) -> None:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise RuntimeError("trajectory expects Nx8 columns: stamp,x,y,z,qx,qy,qz,qw")

    fields = ["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]
    if frame_id is not None:
        fields.append("frame_id")
    if child_frame_id is not None:
        fields.append("child_frame_id")

    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for stamp, x, y, z, qx, qy, qz, qw in trajectory:
            row = [
                f"{float(stamp):.9f}",
                f"{x:.12g}",
                f"{y:.12g}",
                f"{z:.12g}",
                f"{qw:.12g}",
                f"{qx:.12g}",
                f"{qy:.12g}",
                f"{qz:.12g}",
            ]
            if frame_id is not None:
                row.append(frame_id)
            if child_frame_id is not None:
                row.append(child_frame_id)
            writer.writerow(row)


def _joint_transform(origin: Any | None) -> np.ndarray:
    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if origin is not None:
        if "xyz" in origin.attrib:
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=np.float64)
        if "rpy" in origin.attrib:
            rpy = np.fromstring(origin.attrib["rpy"], sep=" ", dtype=np.float64)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _frame_name(frame: Any) -> str:
    return str(frame).strip().lstrip("/")


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


__all__ = [
    "UrdfTransforms",
    "pointcloud_to_numpy",
    "write_trajectory_csv",
]

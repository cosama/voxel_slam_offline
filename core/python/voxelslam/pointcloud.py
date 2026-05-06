from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


_POINT_RECORD_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
    ]
)
_COUNT_DTYPE = np.dtype("<u8")


class PointCloudBuffer:
    """Nx5 point buffer that starts in memory and spills to disk if needed."""

    def __init__(
        self,
        directory: str | Path,
        *,
        memory_limit_bytes: int | None = 4 * 1024**3,
    ) -> None:
        if memory_limit_bytes is not None and memory_limit_bytes < 0:
            raise ValueError("memory_limit_bytes must be non-negative or None")
        self.directory = Path(directory)
        self.memory_limit_bytes = memory_limit_bytes
        self.storage = "memory"
        self.chunks = 0
        self.points = 0
        self.path: Path | None = None
        self._memory_bytes = 0
        self._records: list[np.ndarray] = []
        self._stream = None

    @property
    def byte_size(self) -> int:
        if self._stream is None:
            return self._memory_bytes
        self._stream.flush()
        return self.path.stat().st_size if self.path is not None else 0

    def append(self, points: np.ndarray) -> None:
        records = _points_to_records(points)
        serialized_bytes = records.nbytes + _COUNT_DTYPE.itemsize
        if (
            self._stream is None
            and self.memory_limit_bytes is not None
            and self._memory_bytes + serialized_bytes > self.memory_limit_bytes
        ):
            self._spill_to_disk()

        if self._stream is None:
            self._records.append(records)
            self._memory_bytes += serialized_bytes
        else:
            self._write(records)
        self.chunks += 1
        self.points += int(records.shape[0])

    def iter_points(self):
        if self._stream is None:
            for records in self._records:
                yield _records_to_points(records)
            return

        self._stream.flush()
        self._stream.seek(0)
        while True:
            count = np.fromfile(self._stream, dtype=_COUNT_DTYPE, count=1)
            if count.size == 0:
                break
            records = np.fromfile(
                self._stream,
                dtype=_POINT_RECORD_DTYPE,
                count=int(count[0]),
            )
            if records.shape[0] != int(count[0]):
                raise RuntimeError("truncated pointcloud buffer")
            yield _records_to_points(records)

    def close(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.close()

    def remove(self) -> None:
        self.close()
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def _spill_to_disk(self) -> None:
        stream = tempfile.NamedTemporaryFile(
            prefix="pointcloud_",
            suffix=".bin",
            dir=self.directory,
            delete=False,
        )
        self.path = Path(stream.name)
        self._stream = stream
        self.storage = "disk"
        for records in self._records:
            self._write(records)
        self._records.clear()
        self._memory_bytes = 0

    def _write(self, records: np.ndarray) -> None:
        np.asarray([records.shape[0]], dtype=_COUNT_DTYPE).tofile(self._stream)
        records.tofile(self._stream)


def _points_to_records(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 5:
        raise RuntimeError("pointcloud expects Nx5 columns: timestamp,x,y,z,intensity")
    points = points[np.isfinite(points).all(axis=1)]
    records = np.empty(points.shape[0], dtype=_POINT_RECORD_DTYPE)
    records["timestamp"] = points[:, 0]
    records["x"] = points[:, 1]
    records["y"] = points[:, 2]
    records["z"] = points[:, 3]
    records["intensity"] = points[:, 4]
    return records


def _records_to_points(records: np.ndarray) -> np.ndarray:
    points = np.empty((records.shape[0], 5), dtype=np.float64)
    points[:, 0] = records["timestamp"]
    points[:, 1] = records["x"]
    points[:, 2] = records["y"]
    points[:, 3] = records["z"]
    points[:, 4] = records["intensity"]
    return points


__all__ = ["PointCloudBuffer"]

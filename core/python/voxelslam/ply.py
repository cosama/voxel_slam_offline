from __future__ import annotations

from pathlib import Path

import numpy as np

from .pointcloud import _points_to_records


class BinaryPlyWriter:
    _count_width = 20

    def __init__(self, path: Path):
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

    def __enter__(self) -> "BinaryPlyWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def write(self, points: np.ndarray) -> None:
        records = _points_to_records(points)
        self._stream.write(records.tobytes(order="C"))
        self.count += int(records.shape[0])

    def close(self) -> None:
        if self._stream.closed:
            return
        self._stream.seek(self._count_offset)
        self._stream.write(f"{self.count:0{self._count_width}d}".encode("ascii"))
        self._stream.close()


__all__ = ["BinaryPlyWriter"]

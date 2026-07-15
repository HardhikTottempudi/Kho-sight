"""Kho-Kho court geometry (KPL rulebook v3.0, rule 1) and camera calibration.

Court coordinate frame (metres):
    x: along the 18 m length, 0 at one end line, 18 at the other.
    y: across the 9 m width, 0 at one side line, 9 at the other.
    Centre line: y = 4.5, running the full length (rule 1.4).
    Free zones: x in [0, 2] and x in [16, 18] (rules 1.2, 1.3).
    Seat slots: 8 positions on the centre line, gap 14/9 m between each and to
    the free-zone lines (rule 4.1.2): x = 2 + i * 14/9 for i = 1..8.
    Cones: at the junction of the centre line and each free-zone marker line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

Point = tuple[float, float]

COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
FREE_ZONE_DEPTH = 2.0
MAIN_AREA_X = (FREE_ZONE_DEPTH, COURT_LENGTH - FREE_ZONE_DEPTH)  # (2, 16)
CENTRE_Y = COURT_WIDTH / 2.0  # 4.5
SEAT_GAP = 14.0 / 9.0  # rule 4.1.2
NUM_SEATS = 8


@dataclass(frozen=True)
class Court:
    """Static court geometry and predicates used by the rule engine."""

    length: float = COURT_LENGTH
    width: float = COURT_WIDTH
    free_zone_depth: float = FREE_ZONE_DEPTH
    centre_y: float = CENTRE_Y

    @property
    def seat_positions(self) -> list[Point]:
        """Centres of the 8 seated-chaser slots on the centre line (rule 4.1.2)."""
        return [
            (self.free_zone_depth + (i + 1) * SEAT_GAP, self.centre_y)
            for i in range(NUM_SEATS)
        ]

    @property
    def cone_positions(self) -> list[Point]:
        """Cones at each free-zone marker line on the centre line (rule 4.4.4)."""
        return [
            (self.free_zone_depth, self.centre_y),
            (self.length - self.free_zone_depth, self.centre_y),
        ]

    @property
    def marker_lines_x(self) -> tuple[float, float]:
        """x of the two free-zone marker lines."""
        return (self.free_zone_depth, self.length - self.free_zone_depth)

    def in_bounds(self, p: Point, margin: float = 0.0) -> bool:
        """Point on or inside the boundary line (rule 1.5)."""
        x, y = p
        return (-margin <= x <= self.length + margin) and (
            -margin <= y <= self.width + margin
        )

    def in_free_zone(self, p: Point) -> bool:
        """Strictly inside a free zone; the marker line is NOT part of it (1.6, 4.4.2)."""
        x, y = p
        if not (0.0 <= y <= self.width):
            return False
        return x < self.free_zone_depth or x > self.length - self.free_zone_depth

    def in_main_area(self, p: Point) -> bool:
        """In the 14 m main playing area, marker lines inclusive (rule 4.4.3)."""
        x, y = p
        return (
            MAIN_AREA_X[0] <= x <= MAIN_AREA_X[1] and 0.0 <= y <= self.width
        )

    def side_of_centre(self, p: Point, eps: float = 0.0) -> int:
        """+1 / -1 for either side of the centre line, 0 within +/- eps of it."""
        d = p[1] - self.centre_y
        if abs(d) <= eps:
            return 0
        return 1 if d > 0 else -1

    def nearest_seat(self, p: Point) -> tuple[int, float]:
        """(seat index 0..7, distance in metres) of the closest seat slot."""
        seats = self.seat_positions
        d = [float(np.hypot(p[0] - sx, p[1] - sy)) for sx, sy in seats]
        i = int(np.argmin(d))
        return i, d[i]

    def nearest_marker_line_x(self, x: float) -> float:
        a, b = self.marker_lines_x
        return a if abs(x - a) <= abs(x - b) else b


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to origin, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1).mean()
    s = np.sqrt(2) / max(d, 1e-9)
    T = np.array(
        [[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :2], T


def estimate_homography(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """Normalised DLT homography mapping src (image px) -> dst (court metres).

    Requires >= 4 non-collinear correspondences. Pure numpy so the rule engine
    and tests carry no OpenCV dependency.
    """
    src_a = np.asarray(src, dtype=np.float64)
    dst_a = np.asarray(dst, dtype=np.float64)
    if src_a.shape[0] < 4 or src_a.shape != dst_a.shape:
        raise ValueError("need >= 4 point correspondences of equal count")

    sn, Ts = _normalise(src_a)
    dn, Td = _normalise(dst_a)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows)
    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


@dataclass
class CourtCalibration:
    """Image->court homography for a fixed camera, with save/load."""

    homography: np.ndarray
    image_size: tuple[int, int] | None = None  # (w, h), informational
    court: Court = field(default_factory=Court)

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[Point],
        court_points: Sequence[Point],
        image_size: tuple[int, int] | None = None,
    ) -> "CourtCalibration":
        return cls(estimate_homography(image_points, court_points), image_size)

    def image_to_court(self, points: Sequence[Point]) -> list[Point]:
        pts = np.asarray(points, dtype=np.float64)
        ones = np.ones((len(pts), 1))
        mapped = (self.homography @ np.hstack([pts, ones]).T).T
        mapped /= mapped[:, 2:3]
        return [(float(x), float(y)) for x, y in mapped[:, :2]]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "homography": self.homography.tolist(),
                    "image_size": self.image_size,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CourtCalibration":
        with open(path) as f:
            data = json.load(f)
        size = tuple(data["image_size"]) if data.get("image_size") else None
        return cls(np.asarray(data["homography"], dtype=np.float64), size)

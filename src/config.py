"""Geometry of the limited-angle sinogram completion problem."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """Parallel-beam acquisition: the full sinogram covers [-90, 90) degrees, only a window around 0 is measured.

    The upper bound of the full range is excluded: the views at -90 and +90 degrees hold the same line integrals
    (mirrored detector), so [-90, 90) is one period of the Radon transform, sampled every degree.
    The detector (128 x 1.6 mm = 204.8 mm) is wider than the diagonal of the 140 mm object, so no view is truncated.
    """

    angle_min_deg: float = -25.0
    angle_max_deg: float = 25.0
    num_views_full: int = 180
    full_angle_min_deg: float = -90.0
    full_angle_max_deg: float = 90.0
    det_col_count: int = 128
    det_pixel_size_mm: float = 1.6
    image_shape: tuple[int, int] = (128, 128)
    image_extent_mm: tuple[tuple[float, float], ...] = ((-70.0, 70.0), (-70.0, 70.0))
    sino_norm: float = 100.0  # line integrals (mm) are divided by this to get O(1) sinograms

    @property
    def full_angles(self) -> np.ndarray:
        """Angles (radians) of the ground-truth sinogram."""
        return np.linspace(np.deg2rad(self.full_angle_min_deg), np.deg2rad(self.full_angle_max_deg),
                           self.num_views_full, endpoint=False)

    @property
    def angle_step_deg(self) -> float:
        return (self.full_angle_max_deg - self.full_angle_min_deg) / self.num_views_full

    @property
    def acquired_view_mask(self) -> np.ndarray:
        """[num_views_full] bool, True inside the acquired window (bounds included)."""
        angles_deg = np.rad2deg(self.full_angles)
        eps = 1e-6
        return (angles_deg >= self.angle_min_deg - eps) & (angles_deg <= self.angle_max_deg + eps)

    @property
    def angles(self) -> np.ndarray:
        """Angles (radians) of the acquired views."""
        return self.full_angles[self.acquired_view_mask]

    @property
    def num_views(self) -> int:
        """Number of acquired views."""
        return int(self.acquired_view_mask.sum())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GeometryConfig:
        data = dict(data)
        # Checkpoints written while a fan-beam geometry existed also store its parameters.
        beam = data.pop("beam", "parallel")
        if beam != "parallel":
            raise ValueError(f"Only the parallel-beam geometry is supported, got beam={beam!r}.")
        known = {field.name for field in fields(cls)}
        data = {key: value for key, value in data.items() if key in known}
        data["image_shape"] = tuple(data["image_shape"])
        data["image_extent_mm"] = tuple(tuple(extent) for extent in data["image_extent_mm"])
        return cls(**data)

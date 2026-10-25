"""Parallel-beam projection geometry and the ASTRA operators built on it."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.config import GeometryConfig


class ParallelGeometry:
    """Full-range parallel-beam geometry.

    At angle theta the rays travel along (-sin theta, -cos theta) and the detector axis is
    u(theta) = (cos theta, -sin theta): the sinogram is the Radon transform p(theta, s), s = x . u(theta).
    """

    def __init__(self, angles, det_col_count: int = 128, det_pixel_size: float = 1.6,
                 acquired_window_deg: tuple[float, float] = (-25.0, 25.0)):
        self.angles = np.asarray(angles, dtype=np.float64)
        self.num_views = len(self.angles)
        self.det_col_count = det_col_count
        self.det_pixel_size = det_pixel_size
        self.acquired_window_deg = tuple(acquired_window_deg)

    @classmethod
    def from_config(cls, config: GeometryConfig) -> ParallelGeometry:
        return cls(angles=config.full_angles, det_col_count=config.det_col_count,
                   det_pixel_size=config.det_pixel_size_mm,
                   acquired_window_deg=(config.angle_min_deg, config.angle_max_deg))

    @property
    def angle_step_deg(self) -> float:
        return float(np.rad2deg(np.mean(np.diff(self.angles))))

    @property
    def acquired_view_mask(self) -> np.ndarray:
        """[num_views] bool, True inside the acquired window."""
        angles_deg = np.rad2deg(self.angles)
        lo, hi = self.acquired_window_deg
        eps = 1e-6
        return (angles_deg >= lo - eps) & (angles_deg <= hi + eps)

    def astra_proj_geom(self):
        import astra

        # ASTRA uses ray = (sin a, -cos a) and u = (cos a, sin a): a = -theta gives the convention above.
        return astra.create_proj_geom("parallel", self.det_pixel_size, self.det_col_count, -self.angles)


def astra_vol_geom(config: GeometryConfig):
    import astra

    rows, cols = config.image_shape
    (row_min, row_max), (col_min, col_max) = config.image_extent_mm
    return astra.create_vol_geom(rows, cols, col_min, col_max, row_min, row_max)


def forward_projector(config: GeometryConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    """GPU forward projection (FP_CUDA) of a [H, W] or [1, H, W] image into a [views, detectors] sinogram (mm)."""
    import astra

    proj_geom = ParallelGeometry.from_config(config).astra_proj_geom()
    vol_geom = astra_vol_geom(config)
    projector_id = astra.create_projector("cuda", proj_geom, vol_geom)
    shape = tuple(config.image_shape)

    def project(image: torch.Tensor) -> torch.Tensor:
        array = image.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        array = np.ascontiguousarray(array)
        if array.shape != shape:
            raise ValueError(f"Expected an image of shape {shape}, got {array.shape}")

        vol_id = astra.data2d.create("-vol", vol_geom, array)
        sino_id = astra.data2d.create("-sino", proj_geom)
        cfg = astra.astra_dict("FP_CUDA")
        cfg["ProjectorId"] = projector_id
        cfg["VolumeDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        sinogram = astra.data2d.get(sino_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(vol_id)
        astra.data2d.delete(sino_id)
        return torch.from_numpy(np.asarray(sinogram, dtype=np.float32)).float()

    return project


def fbp_reconstruction(sinogram: np.ndarray, config: GeometryConfig) -> np.ndarray:
    """Filtered back-projection (ram-lak, FBP_CUDA) of a full-range [views, detectors] sinogram in mm."""
    import astra

    proj_geom = ParallelGeometry.from_config(config).astra_proj_geom()
    vol_geom = astra_vol_geom(config)
    projector_id = astra.create_projector("cuda", proj_geom, vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, np.ascontiguousarray(sinogram.astype(np.float32)))
    reco_id = astra.data2d.create("-vol", vol_geom)
    try:
        cfg = astra.astra_dict("FBP_CUDA")
        cfg["ProjectorId"] = projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = reco_id
        cfg["option"] = {"FilterType": "ram-lak"}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        reconstruction = np.asarray(astra.data2d.get(reco_id))
        astra.algorithm.delete(alg_id)
    finally:
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(projector_id)
    return reconstruction

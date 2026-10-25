"""Sinogram metrics shared by training, evaluation and analysis.

Sinograms are compared in normalised units, with a fixed data range of 1 for PSNR and SSIM (a per-sample range
would make the scores depend on the phantom). The "wedge" metrics are restricted to the missing views, the only
ones a model predicts: the acquired views are copied by data consistency.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity

DATA_RANGE = 1.0
METRIC_KEYS = ("mse", "psnr", "ssim", "mse_wedge", "psnr_wedge", "ssim_wedge")


def ssim(target: np.ndarray, pred: np.ndarray, data_range: float = DATA_RANGE) -> float:
    return float(structural_similarity(target, pred, data_range=data_range))  # type: ignore[arg-type]


def ssim_map(target: np.ndarray, pred: np.ndarray, data_range: float = DATA_RANGE) -> np.ndarray:
    return np.asarray(structural_similarity(target, pred, data_range=data_range, full=True)[1])


def psnr(mse: float) -> float:
    return float(10.0 * np.log10(DATA_RANGE**2 / max(mse, 1e-12)))


def sinogram_metrics(pred: np.ndarray, target: np.ndarray, missing_views: np.ndarray) -> Dict[str, float]:
    """METRIC_KEYS of one [V, D] prediction; ``missing_views`` is a [V] bool mask."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    missing_views = np.asarray(missing_views, dtype=bool)

    sq_err = (pred - target) ** 2
    mse_full = float(sq_err.mean())
    mse_wedge = float(sq_err[missing_views].mean())
    per_pixel_ssim = ssim_map(target, pred)
    return {
        "mse": mse_full,
        "psnr": psnr(mse_full),
        "ssim": float(per_pixel_ssim.mean()),
        "mse_wedge": mse_wedge,
        "psnr_wedge": psnr(mse_wedge),
        "ssim_wedge": float(per_pixel_ssim[missing_views].mean()),
    }


def _wedge_dirichlet_energies(sinogram: np.ndarray, missing_views: np.ndarray) -> Tuple[float, float]:
    """Dirichlet energies of the wedge along the angle and along the detector.

    Along the angle, only pairs of consecutive missing views count (not the pairs across the window border).
    """
    consecutive_missing = missing_views[1:] & missing_views[:-1]
    d_angle = np.diff(sinogram, axis=0)[consecutive_missing]
    d_detector = np.diff(sinogram[missing_views], axis=1)
    return float(np.sum(d_angle**2)), float(np.sum(d_detector**2))


def smoothness_metrics(pred: np.ndarray, target: np.ndarray, missing_views: np.ndarray) -> Dict[str, float]:
    """Oversmoothing indicators: Dirichlet energy of the wedge, prediction / ground truth.

    1: as much variation as the ground truth, < 1: smoother (washed-out wedge), > 1: rougher.
    """
    missing_views = np.asarray(missing_views, dtype=bool)
    pred_angle, pred_detector = _wedge_dirichlet_energies(np.asarray(pred, dtype=np.float64), missing_views)
    true_angle, true_detector = _wedge_dirichlet_energies(np.asarray(target, dtype=np.float64), missing_views)
    return {
        "dirichlet_angle_ratio": pred_angle / max(true_angle, 1e-12),
        "dirichlet_detector_ratio": pred_detector / max(true_detector, 1e-12),
    }


class PerViewMoments:
    """Streaming inter-sample variance of [V, D] sinograms, averaged over the detector.

    For the ground truth this is the lowest per-view MSE a prediction independent of the measurements can reach,
    i.e. the floor of a graph model beyond its angular reach. The sums are shifted by the first sample, so that a
    quantity identical for every sample gets a variance of exactly 0.
    """

    def __init__(self) -> None:
        self.count = 0
        self._shift: Optional[np.ndarray] = None
        self._sum: Optional[np.ndarray] = None
        self._sum_sq: Optional[np.ndarray] = None

    def update(self, sinogram: np.ndarray) -> None:
        x = np.asarray(sinogram, dtype=np.float64)
        if self._shift is None:
            self._shift = x.copy()
            self._sum = np.zeros_like(x)
            self._sum_sq = np.zeros_like(x)
        delta = x - self._shift
        self._sum += delta
        self._sum_sq += delta**2
        self.count += 1

    def per_view_variance(self) -> np.ndarray:
        if self._sum is None or self._sum_sq is None:
            raise ValueError("No sample was accumulated.")
        mean = self._sum / self.count
        variance = np.maximum(self._sum_sq / self.count - mean**2, 0.0)
        return variance.mean(axis=1)

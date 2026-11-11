"""Non-learned references: zero filling, linear interpolation along the angle, and its periodic variant."""
from __future__ import annotations

import numpy as np
import torch


def zero_filling(incomplete_sino: torch.Tensor) -> torch.Tensor:
    """The incomplete sinogram itself: missing views stay at zero."""
    return incomplete_sino.clone()


def linear_interpolation(incomplete_sino: torch.Tensor) -> torch.Tensor:
    """Per detector pixel, linear interpolation along the angle between the acquired (non-zero) views.

    Outside the acquired window np.interp holds the value of the closest acquired view. Shape [B, 1, V, D] or [V, D].
    """
    original_shape = incomplete_sino.shape
    result = incomplete_sino.reshape(-1, 1, *original_shape[-2:]).clone()

    for b in range(result.shape[0]):
        sino = result[b, 0].cpu().numpy()  # [V, D]
        acquired = np.abs(sino).sum(axis=1) > 1e-6
        acquired_indices = np.flatnonzero(acquired)
        missing_indices = np.flatnonzero(~acquired)
        if len(acquired_indices) < 2 or len(missing_indices) == 0:
            continue
        for d in range(sino.shape[1]):
            sino[missing_indices, d] = np.interp(missing_indices, acquired_indices, sino[acquired_indices, d])
        result[b, 0] = torch.from_numpy(sino).to(result.device)

    return result.reshape(original_shape)


def periodic_interpolation(incomplete_sino: torch.Tensor) -> torch.Tensor:
    """Linear interpolation across the missing wedge, using p(theta + 180 deg, s) = p(theta, -s).

    The views must cover exactly a half-turn: view i + V is view i with the detector flipped. Each missing view is
    then interpolated between the two closest acquired views on the circle of angles, instead of holding the edge
    view as linear_interpolation does. Shape [B, 1, V, D] or [V, D].
    """
    original_shape = incomplete_sino.shape
    result = incomplete_sino.reshape(-1, 1, *original_shape[-2:]).clone()

    for b in range(result.shape[0]):
        sino = result[b, 0].cpu().numpy()  # [V, D]
        num_views = sino.shape[0]
        acquired = np.abs(sino).sum(axis=1) > 1e-6
        acquired_indices = np.flatnonzero(acquired)
        missing_indices = np.flatnonzero(~acquired)
        if len(acquired_indices) < 2 or len(missing_indices) == 0:
            continue
        flipped = sino[acquired_indices, ::-1]
        positions = np.concatenate([acquired_indices - num_views, acquired_indices, acquired_indices + num_views])
        values = np.concatenate([flipped, sino[acquired_indices], flipped], axis=0)
        for d in range(sino.shape[1]):
            sino[missing_indices, d] = np.interp(missing_indices, positions, values[:, d])
        result[b, 0] = torch.from_numpy(sino).to(result.device)

    return result.reshape(original_shape)

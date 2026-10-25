"""Non-learned references: zero filling and linear interpolation along the angle."""
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

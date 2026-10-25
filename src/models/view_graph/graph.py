"""View graph of a sinogram (one node per projection angle) and the fixed operators built on it.

The graph only depends on the geometry, so it is stored as dense [V, V] operators: with V = 180 this is exact and
much faster than sparse message passing. Both graph models aggregate with the same weights A_hat:

    GCN (transport "identity"):  m_i = sum_j A_hat[i, j] x_j
    SNN (transport "shift"):     m_i = sum_j A_hat[i, j] S(t_f (theta_i - theta_j)) x_j

where S(delta) translates a detector line by delta and t_f is the trace slope of stalk f.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def build_adjacency(angles_rad, k: int = 12, sigma_deg: Optional[float] = 5.0, graph_type: str = "knn",
                    self_loops: bool = True) -> torch.Tensor:
    """Symmetric weighted adjacency W [V, V], float64.

    Args:
        angles_rad: sorted, uniformly spaced view angles.
        k: even number of neighbours, k/2 on each side. The graph is a path, not a cycle: the views near
            +/-90 degrees have fewer neighbours, so one layer reaches exactly k/2 views everywhere.
        sigma_deg: std (degrees) of the Gaussian weights exp(-(theta_i - theta_j)^2 / (2 sigma^2)); None for binary.
        graph_type: "knn" (banded) or "full" (every pair of views).
        self_loops: add w_ii = 1 (A + I, as in Kipf & Welling).
    """
    angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
    num_views = angles.numel()
    idx = torch.arange(num_views)
    hops = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()

    if graph_type == "knn":
        if k < 2 or k % 2 != 0:
            raise ValueError(f"k must be a positive even number (k/2 neighbours per side), got {k}.")
        connected = (hops <= k // 2) & (hops > 0)
    elif graph_type == "full":
        connected = hops > 0
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Use 'knn' or 'full'.")

    if sigma_deg is None:
        weights = torch.ones(num_views, num_views, dtype=torch.float64)
    else:
        delta = angles.unsqueeze(1) - angles.unsqueeze(0)
        weights = torch.exp(-(delta**2) / (2.0 * np.deg2rad(sigma_deg) ** 2))

    adjacency = weights * connected
    if self_loops:
        adjacency = adjacency + torch.eye(num_views, dtype=torch.float64)
    return adjacency


def normalize_adjacency(adjacency: torch.Tensor, normalization: str = "sym") -> torch.Tensor:
    """"sym": D^-1/2 W D^-1/2 (GCN), "rw": D^-1 W (weighted mean)."""
    degree = adjacency.sum(dim=1)
    if normalization == "sym":
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(1) * adjacency * inv_sqrt.unsqueeze(0)
    if normalization == "rw":
        return adjacency / degree.unsqueeze(1)
    raise ValueError(f"Unknown normalization: {normalization}. Use 'sym' or 'rw'.")


def normalized_adjacency(angles_rad, k: int = 12, sigma_deg: Optional[float] = 5.0, graph_type: str = "knn",
                         normalization: str = "sym") -> torch.Tensor:
    """A_hat [V, V] float32: the aggregation weights shared by the GCN and the SNN."""
    return normalize_adjacency(build_adjacency(angles_rad, k, sigma_deg, graph_type), normalization).float()


def shift_slopes(num_stalks: int, max_depth_mm: float) -> np.ndarray:
    """Trace slopes ds/dtheta (mm/rad) of the shift transport, one per stalk, evenly spaced in +/-max_depth_mm.

    In parallel beam, a point at depth t along the rays moves on the detector at |ds/dtheta| = |t|. With
    max_depth_mm the half-diagonal of the image, every point has a stalk whose slope matches its trace.
    """
    return np.linspace(-max_depth_mm, max_depth_mm, num_stalks)


def shift_padding(num_pixels: int, max_shift_px: float) -> int:
    """Samples appended to each detector line before the FFT of the shift transport.

    At least twice the largest translation, so that what enters the detector comes from the padding next to the
    edge it crosses. The total length is odd: the real FFT then has no Nyquist bin and a fractional translation
    stays exactly invertible.
    """
    padding = 2 * math.ceil(max_shift_px) + 2
    return padding + 1 if (num_pixels + padding) % 2 == 0 else padding


def build_shift_phases(angles_rad, slopes_mm_per_rad, pixel_size_mm: float, padded_length: int) -> torch.Tensor:
    """Fourier phases of the shift transport, [F, V, W, 2] float32 (real, imaginary), W = padded_length // 2 + 1.

        phase[f, v, w] = exp(i omega_w t_f theta_v / pixel),   omega_w = 2 pi w / padded_length

    The translation of an edge j -> i factorises as S(t_f theta_i) S(-t_f theta_j): multiplying the spectrum of
    view j by phase[:, j] and the aggregated spectrum of view i by conj(phase[:, i]) applies it exactly (circularly
    on the padded line). It is a flat connection, and R_ji = R_ij^T since a circular translation is orthogonal.
    """
    angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
    slopes = torch.as_tensor(np.asarray(slopes_mm_per_rad), dtype=torch.float64)
    omega = 2.0 * np.pi * torch.arange(padded_length // 2 + 1, dtype=torch.float64) / padded_length
    shift_px = slopes[:, None] * angles[None, :] / pixel_size_mm  # [F, V]
    phase_angle = omega[None, None, :] * shift_px[:, :, None]     # [F, V, W], up to ~300 rad: keep float64 here
    return torch.stack([torch.cos(phase_angle), torch.sin(phase_angle)], dim=-1).float()


def angular_reach_deg(num_layers: int, k: int, angle_step_deg: float, graph_type: str = "knn") -> float:
    """Angular distance over which information can travel: num_layers * k/2 views.

    Every other operation acts within a view, so a measurement cannot influence a missing view further away.
    """
    if graph_type == "full":
        return float("inf")
    return num_layers * (k // 2) * angle_step_deg

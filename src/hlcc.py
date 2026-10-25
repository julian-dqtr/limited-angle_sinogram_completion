"""Helgason-Ludwig consistency conditions (HLCC), in the Chebyshev form of Huang et al. (2017).

Y. Huang et al., "Restoration of missing data in limited angle tomography based on Helgason-Ludwig
consistency conditions", Biomed. Phys. Eng. Express, 2017.

For a parallel-beam sinogram p(theta, s) on a detector |s| <= S, the moment curve of order n is

    a_n(theta) = int p(theta, s) U_n(s / S) ds / S        (U_n: Chebyshev polynomial of the second kind)

and the HLCC state that a_n lies in H_n = span{cos(m theta), sin(m theta) : m <= n, m = n mod 2} (dim n + 1).
They are used twice: as a training penalty (src/losses.py) and as a post-processing that fits a_n on the
acquired views and imposes it on the missing ones (HLCCMomentProjection).
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def detector_coordinate(geom) -> torch.Tensor:
    """Normalised pixel centres s / S in (-1, 1), [D] float64."""
    num_det = geom.det_col_count
    return (torch.arange(num_det, dtype=torch.float64) - num_det / 2.0 + 0.5) * (2.0 / num_det)


def chebyshev_u(x: torch.Tensor, max_order: int) -> torch.Tensor:
    """U_0..U_max_order at x: [len(x), max_order + 1]."""
    polys = [torch.ones_like(x), 2.0 * x]
    for _ in range(2, max_order + 1):
        polys.append(2.0 * x * polys[-1] - polys[-2])
    return torch.stack(polys[: max_order + 1], dim=1)


def moment_operator(geom, max_order: int) -> torch.Tensor:
    """[D, max_order + 1] float64 matrix M such that ``sinogram @ M`` gives the moment curves a_n(theta)."""
    x = detector_coordinate(geom)
    return chebyshev_u(x, max_order) * (2.0 / geom.det_col_count)  # ds / S = 2 / D


def harmonic_basis(angles, order: int) -> torch.Tensor:
    """Basis of H_order sampled at ``angles`` (radians): [V, order + 1] float64."""
    theta = torch.as_tensor(np.asarray(angles), dtype=torch.float64)
    columns = []
    for m in range(order % 2, order + 1, 2):
        columns.append(torch.cos(m * theta))
        if m > 0:
            columns.append(torch.sin(m * theta))
    return torch.stack(columns, dim=1)


def residual_projectors(angles, max_order: int) -> torch.Tensor:
    """I - P_n for n = 0..max_order, P_n the orthogonal projector onto H_n: [max_order + 1, V, V] float64."""
    eye = torch.eye(len(angles), dtype=torch.float64)
    projectors = []
    for n in range(max_order + 1):
        basis = harmonic_basis(angles, n)
        projectors.append(eye - basis @ torch.linalg.pinv(basis))
    return torch.stack(projectors)


class HLCCMomentProjection(nn.Module):
    """Moment regression of Huang et al. (2017), applied to a completed sinogram. No trainable parameter.

    For every order n <= max_order, a_n is fitted in H_n on the acquired views of the measured sinogram and
    evaluated on all views (eq. 13-14). Each missing view of the completed sinogram then receives the correction
    sum_n c_n(theta) W(s) U_n(s / S), W = sqrt(1 - (s / S)^2) (eq. 7), with c_n chosen so that its moments are
    exactly the regressed ones. The acquired views are never modified.

    ``support_threshold`` restricts the correction to the shadow of the object (completed view above that
    fraction of its maximum), so that the zero background stays zero; the moments then match only
    approximately. None disables it. The regression from a +/-25 degree window is stable up to order 4
    (scripts/hlcc_order_study.py).

    Shapes: [..., V, D] for both inputs, computed in float64.
    """

    moment_op: torch.Tensor
    synthesis: torch.Tensor
    extrapolation: torch.Tensor
    acquired: torch.Tensor

    def __init__(self, geom, max_order: int = 4, support_threshold: Optional[float] = 0.03):
        super().__init__()
        if max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        self.max_order = max_order
        self.support_threshold = support_threshold

        x = detector_coordinate(geom)
        moment_op = moment_operator(geom, max_order)                             # [D, K+1]
        atoms = torch.sqrt(1.0 - x**2).unsqueeze(1) * chebyshev_u(x, max_order)  # W U_n, [D, K+1]
        # W U_n is only approximately orthogonal to U_m on a discrete detector: invert the Gram matrix.
        synthesis = atoms @ torch.linalg.inv(moment_op.T @ atoms)                # [D, K+1]

        acquired = torch.from_numpy(np.asarray(geom.acquired_view_mask, dtype=bool))
        extrapolation = []
        for n in range(max_order + 1):
            basis = harmonic_basis(geom.angles, n)                               # [V, n+1]
            extrapolation.append(basis @ torch.linalg.pinv(basis[acquired]))     # [V, V_acquired]

        self.register_buffer("moment_op", moment_op, persistent=False)
        self.register_buffer("synthesis", synthesis, persistent=False)
        self.register_buffer("extrapolation", torch.stack(extrapolation), persistent=False)
        self.register_buffer("acquired", acquired, persistent=False)

    def moments(self, sinogram: torch.Tensor) -> torch.Tensor:
        """[..., V, D] -> [..., V, K+1] float64."""
        return sinogram.double() @ self.moment_op

    def regressed_moments(self, measured: torch.Tensor) -> torch.Tensor:
        """Moment curves on all the views, fitted on the acquired views of ``measured``."""
        acquired_moments = self.moments(measured[..., self.acquired, :])        # [..., V_acq, K+1]
        return torch.einsum("nva,...an->...vn", self.extrapolation, acquired_moments)

    def forward(self, completed: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        delta = self.regressed_moments(measured) - self.moments(completed)      # [..., V, K+1]
        correction = delta @ self.synthesis.T                                   # [..., V, D]
        if self.support_threshold is not None:
            signal = completed.double().clamp_min(0.0)
            correction = correction * (signal > self.support_threshold * signal.amax(dim=-1, keepdim=True))
        missing = (~self.acquired).to(correction.dtype).unsqueeze(1)            # [V, 1]
        return (completed.double() + correction * missing).to(completed.dtype)


class HLCCProjectedModel(nn.Module):
    """A completion model followed by HLCCMomentProjection, with the same interface."""

    def __init__(self, model: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]], geom, max_order: int = 4):
        super().__init__()
        self.model = model
        self.projection = HLCCMomentProjection(geom, max_order)

    def forward(self, incomplete: torch.Tensor) -> torch.Tensor:
        return self.projection(self.model(incomplete), incomplete)

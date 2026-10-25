"""Layers of the view-graph models.

Features are [B, C, V, L]: C = 2 * num_stalks channels, V views (the graph nodes), L detector pixels. Operations
within a view are Conv2d with (1, n) kernels, so they never mix views (and without cuDNN this layout is ~5x faster
than a Conv1d over a [B*V, C, L] batch).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.view_graph.graph import build_shift_phases, normalized_adjacency, shift_padding, shift_slopes


class ViewNorm(nn.Module):
    """LayerNorm of each view over its (channels, detector) entries.

    Statistics are never shared between views, so the normalisation leaks nothing along the angle; the affine
    bias keeps constant (missing) views from being zeroed out.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 3), keepdim=True)
        var = x.var(dim=(1, 3), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def detector_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    """3-tap convolution along the detector. Replicate padding: zero padding creates edges on constant rows
    (acquired flag, missing views) that the normalisation would amplify."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), padding_mode="replicate")


class ViewTransport(nn.Module):
    """Fixed angular aggregation m_i = sum_j A_hat[i, j] T_ij x_j, shared by all the layers of a model.

    - "identity" (GCN): T_ij = I.
    - "shift" (SNN): T_ij translates stalk f (channels 2f, 2f+1) along the detector by t_f (theta_i - theta_j),
      aligning neighbouring views on the points of depth t_f before averaging them (see graph.build_shift_phases).

    Nothing is learned and every buffer is rebuilt from the configuration. Only A_hat mixes views, so the angular
    reach is the one of the graph.
    """

    def __init__(self, angles_rad, transport: str, num_stalks: int, k: int = 12, sigma_deg: Optional[float] = 5.0,
                 graph_type: str = "knn", normalization: str = "sym", num_pixels: Optional[int] = None,
                 pixel_size_mm: Optional[float] = None, max_depth_mm: Optional[float] = None):
        super().__init__()
        if transport not in ("identity", "shift"):
            raise ValueError(f"Unknown transport: {transport}. Use 'identity' or 'shift'.")
        self.transport = transport
        self.num_stalks = num_stalks
        self.num_views = len(angles_rad)
        a_hat = normalized_adjacency(angles_rad, k=k, sigma_deg=sigma_deg, graph_type=graph_type,
                                     normalization=normalization)
        self.register_buffer("a_hat", a_hat, persistent=False)
        if transport == "identity":
            return

        if graph_type != "knn":
            raise ValueError("The shift transport is a small-angle alignment: it needs the local knn graph.")
        if num_pixels is None or pixel_size_mm is None or max_depth_mm is None:
            raise ValueError("The shift transport needs num_pixels, pixel_size_mm and max_depth_mm.")
        angles = torch.as_tensor(np.asarray(angles_rad), dtype=torch.float64)
        max_edge_rad = ((angles[:, None] - angles[None, :]).abs() * (a_hat > 0)).max().item()
        self.slopes_mm_per_rad = shift_slopes(num_stalks, max_depth_mm)
        self.max_shift_px = max_depth_mm * max_edge_rad / pixel_size_mm
        self.num_pixels = num_pixels
        self.padded_length = num_pixels + shift_padding(num_pixels, self.max_shift_px)
        padding = self.padded_length - num_pixels
        self.register_buffer("phase", build_shift_phases(angles_rad, self.slopes_mm_per_rad, pixel_size_mm,
                                                         self.padded_length), persistent=False)
        ramp = torch.arange(1, padding + 1, dtype=torch.float32) / (padding + 1)
        self.register_buffer("ramp", ramp, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transport == "identity":
            return torch.einsum("ij,bcjl->bcil", self.a_hat, x)
        return self._shift_and_aggregate(x)

    def _shift_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
        B, C, V, L = x.shape
        if L != self.num_pixels:
            raise ValueError(f"The shift transport was built for {self.num_pixels} detector pixels, got {L}.")
        # A constant line is invariant under translation: aggregate the constants directly and only send
        # x - x[..., :1] through the FFT. A view no measurement has reached then stays exactly constant, instead of
        # carrying FFT rounding (~1e-7) that ViewNorm would amplify into a detector-position signal.
        offset = x[..., :1]
        x = x - offset
        # The padding ramps from the last pixel back to the first: a periodic extension without a jump, so that the
        # FFT does not ring and a circular translation brings in values close to the edge it crosses.
        ramp = x[..., -1:] + (x[..., :1] - x[..., -1:]) * self.ramp
        spectrum = torch.fft.rfft(torch.cat([x, ramp], dim=-1), dim=-1)          # [B, C, V, W]
        phase = torch.view_as_complex(self.phase)                                 # [F, V, W]
        W = spectrum.size(-1)
        # S(-t_f theta_j) on view j, the GCN aggregation (A_hat is real), then S(t_f theta_i) on view i.
        gauged = (spectrum.reshape(B, self.num_stalks, 2, V, W) * phase[None, :, None]).reshape(B, C, V, W)
        mixed = torch.einsum("ij,bcjwr->bciwr", self.a_hat, torch.view_as_real(gauged)).contiguous()
        mixed = torch.view_as_complex(mixed).reshape(B, self.num_stalks, 2, V, W) * phase.conj()[None, :, None]
        moved = torch.fft.irfft(mixed.reshape(B, C, V, W), n=self.padded_length, dim=-1)[..., :L]
        return moved + torch.einsum("ij,bcjl->bcil", self.a_hat, offset)


class ViewGraphConv(nn.Module):
    """m_i = sum_j A_hat[i, j] T_ij x_j (fixed), then out_i = Conv1x1([x_i ; m_i]) (learned).

    GCN and SNN have exactly the same parameters: only the fixed transport differs.
    """

    def __init__(self, num_stalks: int):
        super().__init__()
        self.num_stalks = num_stalks
        self.conv_update = nn.Conv2d(num_stalks * 4, num_stalks * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, transport: ViewTransport) -> torch.Tensor:
        return self.conv_update(torch.cat([x, transport(x)], dim=1))


class ViewGraphBlock(nn.Module):
    """Residual block: [detector conv, ViewNorm, GELU], [ViewGraphConv, ViewNorm, GELU], detector conv.

    Only the graph conv mixes views, so one block reaches k/2 views.
    """

    def __init__(self, num_stalks: int):
        super().__init__()
        channels = num_stalks * 2
        self.conv1 = detector_conv(channels, channels)
        self.norm1 = ViewNorm(channels)
        self.graph_conv = ViewGraphConv(num_stalks)
        self.norm2 = ViewNorm(channels)
        self.conv2 = detector_conv(channels, channels)

    def forward(self, x: torch.Tensor, transport: ViewTransport) -> torch.Tensor:
        res = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.graph_conv(x, transport)))
        return self.conv2(x) + res

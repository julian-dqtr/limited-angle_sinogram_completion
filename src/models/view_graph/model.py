"""View-graph networks: the GCN baseline and the SinoSheavesNN (SNN)."""
from typing import Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import GeometryConfig
from src.geometry import ParallelGeometry
from src.models.data_consistency import apply_data_consistency, data_consistency_mask
from src.models.view_graph.graph import angular_reach_deg
from src.models.view_graph.layers import ViewGraphBlock, ViewNorm, ViewTransport, detector_conv


class ViewGraphNet(nn.Module):
    """Graph network whose nodes are the views; node features keep the detector axis: [B, 2F, V, D].

    The graph, its transport and the data consistency live inside the model: ``completed = model(incomplete)``
    on [B, 1, V, D], so that training and evaluation cannot diverge.

    Args:
        transport: "identity" (GCN) or "shift" (SNN), see ViewTransport.
        num_stalks: number F of stalks (2F channels; with the shift transport, F depths).
        num_layers: depth. The angular reach is num_layers * k/2 views: 36 / 72 / 108 degrees for 6 / 12 / 18
            layers with k = 12, while the farthest missing view is 65 degrees away from the acquired window.
        k, sigma_deg, graph_type, normalization: see graph.build_adjacency.
        grad_checkpoint: trade compute for memory in deep models.
    """

    dc_mask: torch.Tensor
    acquired_flag: torch.Tensor

    def __init__(self, transport: str = "shift", num_stalks: int = 32, num_layers: int = 6, k: int = 12,
                 sigma_deg: Optional[float] = 5.0, graph_type: str = "knn", normalization: str = "sym",
                 data_consistency: bool = True, grad_checkpoint: bool = False,
                 geometry_config: Optional[GeometryConfig] = None):
        super().__init__()
        self.geometry_config = geometry_config or GeometryConfig()
        geom = ParallelGeometry.from_config(self.geometry_config)
        self.transport = transport
        self.num_layers = num_layers
        self.k = k
        self.graph_type = graph_type
        self.data_consistency = data_consistency
        self.grad_checkpoint = grad_checkpoint

        # The shift transport covers every depth of the image, up to its half-diagonal.
        (y_min, y_max), (x_min, x_max) = self.geometry_config.image_extent_mm
        self.transport_op = ViewTransport(
            geom.angles, transport, num_stalks, k=k, sigma_deg=sigma_deg, graph_type=graph_type,
            normalization=normalization, num_pixels=self.geometry_config.det_col_count,
            pixel_size_mm=self.geometry_config.det_pixel_size_mm,
            max_depth_mm=float(np.hypot((y_max - y_min) / 2, (x_max - x_min) / 2)),
        )
        self.register_buffer("dc_mask", data_consistency_mask(geom), persistent=False)
        acquired = torch.from_numpy(geom.acquired_view_mask).float().view(1, 1, -1, 1)
        self.register_buffer("acquired_flag", acquired, persistent=False)

        # Input channels: the measured signal and an "acquired view" flag.
        self.encoder = detector_conv(2, num_stalks * 2)
        self.blocks = nn.ModuleList([ViewGraphBlock(num_stalks) for _ in range(num_layers)])
        hidden = max(1, num_stalks // 2)
        self.decoder = nn.Sequential(
            detector_conv(num_stalks * 2, num_stalks),
            ViewNorm(num_stalks),
            nn.GELU(),
            detector_conv(num_stalks, hidden),
            nn.GELU(),
            detector_conv(hidden, 1),
        )

    @property
    def angular_reach_deg(self) -> float:
        return angular_reach_deg(self.num_layers, self.k, self.geometry_config.angle_step_deg, self.graph_type)

    def forward(self, incomplete: torch.Tensor, apply_dc: Optional[bool] = None) -> torch.Tensor:
        if incomplete.dim() != 4 or incomplete.size(1) != 1:
            raise ValueError(f"Expected input shape [B, 1, V, D], got {tuple(incomplete.shape)}")
        B, _, V, D = incomplete.shape
        if V != self.transport_op.num_views:
            raise ValueError(f"The model graph has {self.transport_op.num_views} views, got a sinogram with {V}.")

        x = torch.cat([incomplete, self.acquired_flag.expand(B, 1, V, D)], dim=1)  # [B, 2, V, D]
        x = F.gelu(self.encoder(x))                                                # [B, 2F, V, D]
        for block in self.blocks:
            if self.grad_checkpoint and self.training and x.requires_grad:
                x = cast(torch.Tensor, checkpoint(block, x, self.transport_op, use_reentrant=False))
            else:
                x = block(x, self.transport_op)
        out = self.decoder(x)                                                      # [B, 1, V, D]

        if self.data_consistency if apply_dc is None else apply_dc:
            out = apply_data_consistency(incomplete, out, self.dc_mask)
        return out


class SinoSheafNet(ViewGraphNet):
    """SNN: hard-coded restriction maps that translate stalk f along the detector by t_f (theta_i - theta_j)."""

    def __init__(self, **kwargs):
        kwargs.pop("transport", None)
        super().__init__(transport="shift", **kwargs)


class SinoGCN(ViewGraphNet):
    """GCN baseline: same graph, layers and parameters as the SNN, with an identity transport."""

    def __init__(self, **kwargs):
        kwargs.pop("transport", None)
        super().__init__(transport="identity", **kwargs)

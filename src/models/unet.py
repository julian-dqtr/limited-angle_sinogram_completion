"""U-Net on the sinogram, used by UNet2D (MSE) and UNet2dHLCC (MSE + HLCC penalty)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GeometryConfig
from src.geometry import ParallelGeometry
from src.models.data_consistency import apply_data_consistency, data_consistency_mask


class ConvBlock(nn.Module):
    """Two [conv 3x3, InstanceNorm, LeakyReLU] layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))


class UpBlock(nn.Module):
    """Bilinear upsampling + conv (no checkerboard artefacts, unlike a transposed conv), then skip concatenation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.conv_up(self.up(x))
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv_block(torch.cat([skip, x], dim=1))


class UNetBackbone(nn.Module):
    """Three-level U-Net. Inputs are padded (replicate) to a multiple of 8 and the output is cropped back."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16):
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be a positive integer")
        self._divisor = 8  # three 2x2 poolings
        f = [filters, filters * 2, filters * 4, filters * 8]

        self.enc1 = ConvBlock(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f[0], f[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(f[1], f[2])
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f[2], f[3])
        self.dec3 = UpBlock(f[3], f[2])
        self.dec2 = UpBlock(f[2], f[1])
        self.dec1 = UpBlock(f[1], f[0])
        self.out_conv = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self._divisor, (-w) % self._divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)[..., :h, :w]


class SinogramUNet(nn.Module):
    """``completed = model(incomplete)`` on [B, 1, V, D]: residual U-Net followed by hard data consistency."""

    dc_mask: torch.Tensor

    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters: int = 16, data_consistency: bool = True,
                 geometry_config: Optional[GeometryConfig] = None) -> None:
        super().__init__()
        self.network = UNetBackbone(in_channels=in_channels, out_channels=out_channels, filters=filters)
        self.data_consistency = data_consistency
        geom = ParallelGeometry.from_config(geometry_config or GeometryConfig())
        # Derived from the geometry stored in the checkpoint, hence not persistent.
        self.register_buffer("dc_mask", data_consistency_mask(geom), persistent=False)

    def forward(self, x: torch.Tensor, apply_dc: Optional[bool] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, V, D], got {tuple(x.shape)}")
        out = x + self.network(x)
        if self.data_consistency if apply_dc is None else apply_dc:
            out = apply_data_consistency(x, out, self.dc_mask)
        return out


class SinogramUNetHLCC(SinogramUNet):
    """The same network as SinogramUNet: UNet2D and UNet2dHLCC only differ by their training loss."""

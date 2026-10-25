"""Plotting style shared by the analysis scripts, and the figures saved at the end of a training run."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.config import GeometryConfig
from src.geometry import fbp_reconstruction

# Colour = model family, so that a model keeps its colour across figures; text always uses the ink tokens.
INK, SECONDARY, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#f0efec"
FAMILY_COLORS = {"Baseline": "#52514e", "U-Net": "#2a78d6", "GCN": "#eb6834", "SNN": "#1baf7a"}


def model_family(model_name: str) -> str:
    from src.models.factory import BASELINES, parse_model_name

    base, _ = parse_model_name(model_name)
    return "Baseline" if base in BASELINES else "U-Net" if base.startswith("UNet") else base


def signed(value: float, digits: int = 1, unit: str = " %") -> str:
    """"+3.1 %" / "−9.8 %", with a typographic minus sign."""
    return f"{value:+.{digits}f}{unit}".replace("-", "−")


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, plt, path: Path, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {path}")


def _extents(config: GeometryConfig):
    (row_min, row_max), (col_min, col_max) = config.image_extent_mm
    half_detector = config.det_col_count * config.det_pixel_size_mm / 2.0
    image = [col_min, col_max, row_min, row_max]
    sinogram = [config.full_angle_min_deg, config.full_angle_max_deg, -half_detector, half_detector]
    return image, sinogram


def _show_sinogram(ax, sinogram: torch.Tensor, title: str, extent) -> None:
    image = sinogram.detach().cpu().squeeze().numpy().T  # angle along x
    ax.imshow(image, cmap="bone", origin="lower", aspect="auto", extent=extent)
    ax.set_title(title)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("detector s (mm)")


def _show_image(ax, image: np.ndarray, title: str, extent) -> None:
    ax.imshow(image, cmap="gray", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def save_dataset_preview(dataset, config: GeometryConfig, figures_dir: Path) -> None:
    """Phantom, full and incomplete sinograms of the first training sample."""
    plt = _pyplot()
    incomplete, full, phantom = dataset[0]
    image_extent, sino_extent = _extents(config)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _show_image(axes[0], phantom.squeeze(0).numpy(), "Phantom (sample 0)", image_extent)
    _show_sinogram(axes[1], full, f"Full sinogram ({config.num_views_full} views)", sino_extent)
    _show_sinogram(axes[2], incomplete, f"Incomplete sinogram ({config.num_views} acquired views)", sino_extent)
    _save(fig, plt, figures_dir / "dataset_preview.png")


def save_training_curve(train_losses: List[float], figures_dir: Path) -> None:
    if not train_losses:
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.grid(True, alpha=0.3)
    _save(fig, plt, figures_dir / "training_losses.png")


def save_example_figure(model, dataset, device, config: GeometryConfig, figures_dir: Path, wandb_run=None,
                        sample_index: int = 0) -> None:
    """Completed sinogram and its FBP for one validation sample. A plotting failure never stops a finished run."""
    try:
        plt = _pyplot()
        model.eval()
        incomplete, full, phantom = dataset[sample_index % len(dataset)]
        with torch.no_grad():
            completed = model(incomplete.unsqueeze(0).to(device)).cpu()
        image_extent, sino_extent = _extents(config)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        _show_image(axes[0, 0], phantom.squeeze(0).numpy(), "Phantom", image_extent)
        _show_sinogram(axes[0, 1], full, "Full sinogram", sino_extent)
        _show_sinogram(axes[0, 2], incomplete, "Incomplete sinogram", sino_extent)
        _show_sinogram(axes[1, 0], completed, "Completed sinogram (model output)", sino_extent)
        reconstruction = fbp_reconstruction(completed.squeeze().numpy() * config.sino_norm, config)
        _show_image(axes[1, 1], reconstruction, "FBP of the completed sinogram", image_extent)
        axes[1, 2].axis("off")
        path = figures_dir / "example_after_training.png"
        _save(fig, plt, path)

        if wandb_run is not None:
            import wandb

            wandb_run.log({"example_reconstruction": wandb.Image(str(path))})
    except Exception as exc:
        traceback.print_exc()
        print(f"Warning: could not save the example figure ({exc}).")

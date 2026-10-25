"""Shared fixtures. Run with: .venv/bin/python -m pytest"""
import numpy as np
import pytest
import torch

from src.config import GeometryConfig
from src.dataset import SinogramCompletionDataset
from src.geometry import ParallelGeometry

torch.backends.cudnn.enabled = False  # Tesla K80


@pytest.fixture(scope="session")
def config():
    return GeometryConfig()


@pytest.fixture(scope="session")
def geom(config):
    return ParallelGeometry.from_config(config)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def clean_val_batch(config):
    """8 noiseless validation samples: (incomplete [8, 1, V, D], full [8, 1, V, D], phantom [8, 1, H, W])."""
    dataset = SinogramCompletionDataset(8, split="val", noise_level=0.0, geometry_config=config)
    items = [dataset[i] for i in range(len(dataset))]
    return tuple(torch.stack([item[j] for item in items]) for j in range(3))


def analytic_disk_sinogram(geom, radius=20.0, center=(15.0, -10.0), density=0.5):
    """Exact parallel-beam sinogram of a uniform disk, integrated in closed form over each detector pixel.

    p(theta, s) = 2 density sqrt(r^2 - (s - c . u(theta))^2), u(theta) = (cos theta, -sin theta). Its moments are
    M0 = density pi r^2 and M1(theta) = M0 (cx cos(theta) - cy sin(theta)).
    """
    n_det, ds = geom.det_col_count, geom.det_pixel_size
    edges = (np.arange(n_det + 1) - n_det / 2.0) * ds
    shift = center[0] * np.cos(geom.angles) - center[1] * np.sin(geom.angles)
    t = np.clip(edges[None, :] - shift[:, None], -radius, radius)
    primitive = t * np.sqrt(radius**2 - t**2) + radius**2 * np.arcsin(t / radius)  # of 2 sqrt(r^2 - t^2)
    return torch.from_numpy(density * np.diff(primitive, axis=1) / ds).float()

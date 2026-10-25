"""Random 2D phantoms and their (incomplete, full) sinogram pairs, generated on the fly."""
from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import GeometryConfig
from src.geometry import forward_projector

PhantomType = Literal["shepp_logan", "ellipses", "blobs", "rectangles", "mixed"]
Split = Literal["train", "val", "test"]

# Sample i of a split is drawn from SeedSequence([SPLIT_STREAMS[split], base_seed, i]): the splits are disjoint,
# fixed, and independent of the global RNG state.
SPLIT_STREAMS = {"train": 0, "val": 1, "test": 2}


# Every shape generator draws from an explicit Generator, so that a phantom is a pure function of its seed.
def _grid(shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = shape
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, H), torch.linspace(-1.0, 1.0, W), indexing="ij")
    return yy, xx


def _rotate(xx, yy, cx: float, cy: float, angle_deg: float):
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    x_rot = cos_a * (xx - cx) + sin_a * (yy - cy)
    y_rot = -sin_a * (xx - cx) + cos_a * (yy - cy)
    return x_rot, y_rot


def _random_ellipses(shape: Tuple[int, int], n: int, rng: np.random.Generator) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.float32)
    yy, xx = _grid(shape)
    for _ in range(n):
        cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
        rx, ry = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5)
        angle, intensity = rng.uniform(0, 180), rng.uniform(0.1, 1.0)
        x_rot, y_rot = _rotate(xx, yy, cx, cy, angle)
        mask[(x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0] += intensity
    return torch.clamp(mask, 0.0, 1.0)


def _random_blobs(shape: Tuple[int, int], n: int, rng: np.random.Generator) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.float32)
    yy, xx = _grid(shape)
    for _ in range(n):
        cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
        sigma, intensity = rng.uniform(0.05, 0.2), rng.uniform(0.1, 1.0)
        mask += intensity * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return torch.clamp(mask, 0.0, 1.0)


def _random_rectangles(shape: Tuple[int, int], n: int, rng: np.random.Generator) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.float32)
    yy, xx = _grid(shape)
    for _ in range(n):
        cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
        w, h = rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6)
        angle, intensity = rng.uniform(0, 180), rng.uniform(0.1, 1.0)
        x_rot, y_rot = _rotate(xx, yy, cx, cy, angle)
        mask[(x_rot.abs() <= w / 2) & (y_rot.abs() <= h / 2)] += intensity
    return torch.clamp(mask, 0.0, 1.0)


def _shepp_logan(shape: Tuple[int, int]) -> torch.Tensor:
    from skimage.data import shepp_logan_phantom

    image = torch.from_numpy(shepp_logan_phantom().astype(np.float32))[None, None]
    return F.interpolate(image, size=shape, mode="bilinear", align_corners=False)[0, 0]


def make_phantom(shape: Tuple[int, int], phantom_type: PhantomType, rng: np.random.Generator) -> torch.Tensor:
    """[H, W] phantom in [0, 1], with a random rotation, scale and shift."""
    if phantom_type == "mixed":
        phantom_type = ("ellipses", "blobs", "rectangles", "shepp_logan")[rng.integers(4)]

    if phantom_type == "shepp_logan":
        base = _shepp_logan(shape)
    elif phantom_type == "ellipses":
        base = _random_ellipses(shape, int(rng.integers(2, 6)), rng)
    elif phantom_type == "blobs":
        base = _random_blobs(shape, int(rng.integers(3, 8)), rng)
    elif phantom_type == "rectangles":
        base = _random_rectangles(shape, int(rng.integers(2, 5)), rng)
    else:
        raise ValueError(f"Unknown phantom_type: {phantom_type}")

    rad = np.deg2rad(rng.uniform(-180, 180))
    scale = rng.uniform(0.85, 1.0)
    tx, ty = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    theta = torch.eye(2, 3)
    theta[0, 0], theta[0, 1], theta[0, 2] = scale * cos_a, scale * -sin_a, tx
    theta[1, 0], theta[1, 1], theta[1, 2] = scale * sin_a, scale * cos_a, ty

    grid = F.affine_grid(theta.unsqueeze(0), [1, 1, *shape], align_corners=False)
    base = F.grid_sample(base[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
    return torch.clamp(base, 0.0, 1.0)


class SinogramCompletionDataset(Dataset):
    """(incomplete [1, V, D], full [1, V, D], phantom [1, H, W]) triplets, sinograms divided by ``sino_norm``.

    Args:
        n_samples: size of the split.
        split: "train", "val" or "test".
        noise_level: incident photon count I0 of the Poisson noise (<= 0 disables it).
        base_seed: changes every split at once.
        fixed_train_set: if False, the training split draws a new phantom at every access.
        noisy_target: if True, the target carries the same noise as the input (by default it is clean,
            which is the sinogram that satisfies the consistency conditions).
    """

    def __init__(
        self,
        n_samples: int,
        split: Split = "train",
        geometry_config: Optional[GeometryConfig] = None,
        noise_level: float = 1e5,
        phantom_type: PhantomType = "mixed",
        base_seed: int = 0,
        fixed_train_set: bool = True,
        noisy_target: bool = False,
        projector_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if split not in SPLIT_STREAMS:
            raise ValueError(f"Unknown split '{split}'. Use one of {tuple(SPLIT_STREAMS)}.")
        self.n_samples = n_samples
        self.split = split
        self.geometry_config = geometry_config or GeometryConfig()
        self.noise_level = noise_level
        self.phantom_type: PhantomType = phantom_type
        self.base_seed = int(base_seed)
        self.fixed_train_set = fixed_train_set
        self.noisy_target = noisy_target
        # No fallback: a broken ASTRA install must stop the run, not train on a fake projector.
        self.projector_fn = projector_fn or forward_projector(self.geometry_config)
        self.acquired_views = torch.from_numpy(np.flatnonzero(self.geometry_config.acquired_view_mask)).long()

    @property
    def is_deterministic(self) -> bool:
        return self.split != "train" or self.fixed_train_set

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.n_samples:
            raise IndexError(index)

        rng = self._sample_rng(index)
        phantom = make_phantom(self.geometry_config.image_shape, self.phantom_type, rng)
        clean = self.projector_fn(phantom).cpu()
        noise_generator = torch.Generator().manual_seed(int(rng.integers(0, 2**62)))
        noisy = self._add_poisson_noise(clean, noise_generator)

        norm = self.geometry_config.sino_norm
        full = (noisy if self.noisy_target else clean) / norm
        incomplete = self._keep_acquired_views(noisy / norm)
        return incomplete.unsqueeze(0), full.unsqueeze(0), phantom.unsqueeze(0)

    def _sample_rng(self, index: int) -> np.random.Generator:
        if not self.is_deterministic:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([SPLIT_STREAMS[self.split], self.base_seed, int(index)]))

    def _add_poisson_noise(self, sinogram: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Poisson noise on a raw sinogram (mm). Line integrals are scaled by 0.1 before the exponential,
        so that the largest attenuations (~80 mm -> 8) stay physically plausible."""
        if self.noise_level <= 0.0:
            return sinogram
        i0, scale = self.noise_level, 0.1
        photons = torch.poisson(i0 * torch.exp(-sinogram * scale), generator=generator)
        return -torch.log(torch.clamp(photons, min=1.0) / i0) / scale

    def _keep_acquired_views(self, sinogram: torch.Tensor) -> torch.Tensor:
        incomplete = torch.zeros_like(sinogram)
        incomplete[self.acquired_views] = sinogram[self.acquired_views]
        return incomplete

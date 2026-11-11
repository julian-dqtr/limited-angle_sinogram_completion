"""Non-learned baselines, in particular the periodic interpolation across the missing wedge."""
import torch

from src.models.baselines import linear_interpolation, periodic_interpolation
from src.models.factory import get_model


def _missing(geom) -> torch.Tensor:
    return torch.from_numpy(~geom.acquired_view_mask)


def test_periodic_interpolation_uses_the_half_turn_symmetry(geom, clean_val_batch):
    """On a clean sinogram, the completed view 0 (-90 deg) is one angular step away from the completed view V-1
    (+89 deg) with the detector flipped: the periodic extension p(theta + 180, s) = p(theta, -s) is continuous."""
    incomplete, full, _ = clean_val_batch
    out = periodic_interpolation(incomplete)
    assert out.shape == incomplete.shape and out.dtype == incomplete.dtype
    first, last_flipped = out[:, :, 0], out[:, :, -1].flip(-1)
    gap = (first - last_flipped).abs().amax().item()
    step = (out[:, :, 1:] - out[:, :, :-1]).abs().amax().item()  # largest change between adjacent views
    assert gap <= step
    # Linear interpolation (np.interp holds the edge view) is not continuous across the half turn.
    linear = linear_interpolation(incomplete)
    assert (linear[:, :, 0] - linear[:, :, -1].flip(-1)).abs().amax().item() > gap


def test_periodic_interpolation_keeps_the_measured_views(geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    missing = _missing(geom)
    out = periodic_interpolation(incomplete)
    assert torch.equal(out[:, :, ~missing], incomplete[:, :, ~missing])
    assert not torch.equal(out[:, :, missing], incomplete[:, :, missing])
    # A complete sinogram is left untouched, and the [V, D] shape is accepted.
    assert torch.equal(periodic_interpolation(full), full)
    assert torch.equal(periodic_interpolation(incomplete[0, 0]), out[0, 0])


def test_periodic_interpolation_beats_linear_interpolation_on_the_wedge(geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    missing = _missing(geom)

    def wedge_mse(sinogram):
        return ((sinogram - full)[:, :, missing] ** 2).mean().item()

    assert wedge_mse(periodic_interpolation(incomplete)) < wedge_mse(linear_interpolation(incomplete))


def test_periodic_interp_from_the_factory(config, geom, clean_val_batch, device):
    incomplete = clean_val_batch[0][:2].to(device)
    missing = _missing(geom)
    model, is_nn = get_model("PeriodicInterp", device, geometry_config=config)
    assert not is_nn
    assert torch.equal(model(incomplete), periodic_interpolation(incomplete))
    projected, is_nn = get_model("PeriodicInterp_P4", device, geometry_config=config)
    out = projected(incomplete)
    assert not is_nn and sum(p.numel() for p in projected.parameters()) == 0
    assert torch.equal(out[:, :, ~missing], incomplete[:, :, ~missing])
    assert not torch.equal(out, model(incomplete))

"""Hard data consistency: the measured views come out unchanged, the missing views are never touched."""
import numpy as np
import torch

from src.models.data_consistency import apply_data_consistency, data_consistency_mask


def test_mask_is_the_acquired_view_mask(geom):
    mask = data_consistency_mask(geom)
    assert mask.shape == (1, 1, geom.num_views, 1)
    np.testing.assert_array_equal(mask.flatten().numpy(), geom.acquired_view_mask.astype(np.float32))


def test_dc_is_the_identity_on_the_ground_truth(geom, clean_val_batch):
    incomplete, full, _ = clean_val_batch
    torch.testing.assert_close(apply_data_consistency(incomplete, full, data_consistency_mask(geom)), full,
                               atol=1e-7, rtol=0)


def test_dc_returns_every_measured_view_exactly(geom, clean_val_batch):
    incomplete, _, _ = clean_val_batch
    prediction = torch.randn(incomplete.shape, generator=torch.Generator().manual_seed(0))
    out = apply_data_consistency(incomplete, prediction, data_consistency_mask(geom))
    acquired = torch.from_numpy(geom.acquired_view_mask)
    assert torch.equal(out[:, :, acquired], incomplete[:, :, acquired])
    assert torch.equal(out[:, :, ~acquired], prediction[:, :, ~acquired])

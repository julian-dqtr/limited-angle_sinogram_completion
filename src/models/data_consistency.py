"""Hard data consistency: every model copies the measured views into its output."""
import numpy as np
import torch


def data_consistency_mask(geom) -> torch.Tensor:
    """[1, 1, V, 1] float32 mask, 1 on the acquired views and 0 on the missing ones."""
    mask = torch.from_numpy(np.asarray(geom.acquired_view_mask, dtype=np.float32))
    return mask.view(1, 1, geom.num_views, 1)


def apply_data_consistency(measured: torch.Tensor, predicted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask * measured + (1.0 - mask) * predicted

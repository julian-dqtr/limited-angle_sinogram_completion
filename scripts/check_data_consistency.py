"""Check that the acquired views of the completed sinograms are exactly the measured ones.

    python -m scripts.check_data_consistency
    python -m scripts.check_data_consistency --models UNet2D_N200 UNet2D_N200_P4 --num_samples 200

The input is noisy and the ground truth clean: the acquired views of the output must equal the input, so every
difference printed here should be exactly 0.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.config import GeometryConfig
from src.dataset import SinogramCompletionDataset
from src.geometry import ParallelGeometry
from src.models.factory import get_model

torch.backends.cudnn.enabled = False  # Tesla K80


def acquired_difference(output: torch.Tensor, incomplete: torch.Tensor, acquired: np.ndarray) -> dict:
    """|output - input| on the acquired views of [N, 1, V, D] tensors."""
    difference = (output - incomplete)[:, 0, acquired].abs()
    per_view = difference.amax(dim=(0, 2))
    return {"max |output - input|": difference.max().item(), "mean |output - input|": difference.mean().item(),
            "views that differ": f"{int((per_view > 0).sum())} / {int(acquired.sum())}"}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Difference between the measured views and the model outputs")
    parser.add_argument("--models", nargs="+", default=["UNet2D", "UNet2dHLCC"])
    parser.add_argument("--num_samples", type=int, default=100, help="Size of the test split")
    parser.add_argument("--noise_level", type=float, default=1e5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GeometryConfig()
    geom = ParallelGeometry.from_config(config)
    acquired = np.asarray(geom.acquired_view_mask, dtype=bool)
    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level,
                                        geometry_config=config)
    samples = [dataset[i] for i in range(len(dataset))]
    incomplete = torch.stack([s[0] for s in samples])  # noisy measurements
    full = torch.stack([s[1] for s in samples])        # clean target

    rows = []
    for name in args.models:
        try:
            model, _ = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[NOT EVALUATED] {name}: {str(exc).splitlines()[0]}")
            continue
        output = torch.cat([model(incomplete[i:i + 25].to(device)).float().cpu()
                            for i in range(0, len(incomplete), 25)])
        rows.append({"model": name, **acquired_difference(output, incomplete, acquired)})

    signal = incomplete[:, 0, acquired]
    noise = (incomplete - full)[:, 0, acquired].abs()
    print(f"\n{len(dataset)} test samples, {int(acquired.sum())} acquired views out of {geom.num_views}. "
          f"Measured values: mean {signal.mean():.3f}, max {signal.max():.3f}.")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3e}"))
    print(f"\nFor scale, the noise of the input (|input - ground truth| on the acquired views): "
          f"mean {noise.mean():.3e}, max {noise.max():.3e}.")


if __name__ == "__main__":
    main()

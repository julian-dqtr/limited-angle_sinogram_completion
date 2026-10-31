"""Moment curves a_n(theta) of orders 0 to 4: ground truth, HLCC regression from the window and models (RQ2).

    python -m scripts.plot_moment_curves
    python -m scripts.plot_moment_curves --models UNet2D UNet2D_N200 --sample 12

Top row: the curves of one test sample on the 180 views (by default the sample of median wedge MSE of the first
model): the ground truth, the regression fitted on the 51 measured views and extended to the missing ones (src/hlcc.py,
the moments that the _P4 models impose) and the output of each model. Bottom row: the error of each curve per view
over the test split, relative to the RMS of the true curve of that order.

Outputs: results/figures/hlcc_moment_curves.pdf and results/tables/hlcc_moment_curves.csv (error of the curves on the
missing views per order, pooled as in scripts/hlcc_order_study.py).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.hlcc_order_study import moment_error, predict, save, series_styles
from src.config import GeometryConfig
from src.dataset import SinogramCompletionDataset
from src.geometry import ParallelGeometry
from src.hlcc import HLCCMomentProjection
from src.models.factory import get_model
from src.paths import RESULTS_DIR
from src.plotting import BAND, GRID, INK, SECONDARY, style_axis

torch.backends.cudnn.enabled = False  # Tesla K80

REGRESSION_COLOR = "#4a3aa7"  # violet, as the _P4 curves of results/model_analysis.ipynb
REGRESSION = "HLCC regression from the measured views"
ANGLE_TICKS = [-90, -25, 0, 25, 90]


def per_view_error(moments: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    """RMS error over the samples per view, relative to the RMS of the true curve of each order: [V, K+1]."""
    error = (moments - reference).pow(2).mean(dim=0).sqrt()
    return (error / reference.pow(2).mean(dim=(0, 1)).sqrt()).numpy()


def angle_axis(ax, window_edges) -> None:
    ax.axvspan(*window_edges, color=BAND, zorder=0)
    ax.grid(color=GRID, linewidth=0.8)
    style_axis(ax)
    ax.set_xticks(ANGLE_TICKS)
    ax.set_xlim(-90, 90)


def main():
    parser = argparse.ArgumentParser(description="Moment curves of orders 0 to K: ground truth, regression, models")
    parser.add_argument("--models", nargs="+", default=["UNet2D"])
    parser.add_argument("--num_samples", type=int, default=200, help="Size of the test split")
    parser.add_argument("--max_order", type=int, default=4)
    parser.add_argument("--sample", type=int, default=None, help="Test sample of the top row (default: median)")
    parser.add_argument("--noise_level", type=float, default=1e5)
    parser.add_argument("--out_figure", type=Path, default=RESULTS_DIR / "figures" / "hlcc_moment_curves.pdf")
    parser.add_argument("--out_csv", type=Path, default=RESULTS_DIR / "tables" / "hlcc_moment_curves.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GeometryConfig()
    geom = ParallelGeometry.from_config(config)
    missing = torch.from_numpy(~geom.acquired_view_mask)
    orders = list(range(args.max_order + 1))
    angles_deg = np.rad2deg(geom.angles)
    step = config.angle_step_deg
    window_edges = (config.angle_min_deg - step / 2, config.angle_max_deg + step / 2)

    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level,
                                        geometry_config=config)
    samples = [dataset[i] for i in range(len(dataset))]
    incomplete = torch.stack([s[0] for s in samples])  # noisy measurements, [N, 1, V, D]
    full = torch.stack([s[1] for s in samples])        # clean target

    predictions = {}
    for name in args.models:
        try:
            model, _ = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[NOT EVALUATED] {name}: {str(exc).splitlines()[0]}")
            continue
        predictions[name] = predict(model, incomplete, device)
    if not predictions:
        raise SystemExit("No model could be loaded: train them first (src.train).")
    styles = series_styles(predictions)

    projection = HLCCMomentProjection(geom, args.max_order)
    true_moments = projection.moments(full)                   # [N, 1, V, K+1]
    regressed = projection.regressed_moments(incomplete)
    model_moments = {name: projection.moments(pred) for name, pred in predictions.items()}

    sample = args.sample
    if sample is None:
        first = predictions[next(iter(predictions))]
        wedge_mse = ((first - full)[:, 0][:, missing] ** 2).mean(dim=(1, 2)).numpy()
        sample = int(np.argsort(wedge_mse)[len(wedge_mse) // 2])

    # Error of the curves on the missing views, per order (the panel (b) of scripts/hlcc_order_study.py)
    true_missing = true_moments[:, :, missing]
    table = pd.DataFrame({"order": orders,
                          "moment_error_regression": moment_error(regressed[:, :, missing], true_missing)})
    for name, moments in model_moments.items():
        table[f"moment_error_{name}"] = moment_error(moments[:, :, missing], true_missing)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"{len(dataset)} test samples, relative error of the moment curves on the missing views:")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3g}"))

    true_error = {REGRESSION: per_view_error(regressed[:, 0], true_moments[:, 0]),
                  **{name: per_view_error(m[:, 0], true_moments[:, 0]) for name, m in model_moments.items()}}
    fig, axes = plt.subplots(2, len(orders), figsize=(3.3 * len(orders), 6.0), sharex=True, squeeze=False,
                             layout="constrained")
    for n in orders:
        ax_curve, ax_error = axes[0, n], axes[1, n]
        angle_axis(ax_curve, window_edges)
        ax_curve.plot(angles_deg, true_moments[sample, 0, :, n].numpy(), color=INK, linewidth=2.2, label="ground truth")
        for name, moments in model_moments.items():
            ax_curve.plot(angles_deg, moments[sample, 0, :, n].numpy(), linewidth=1.6, label=name,
                          color=styles[name]["color"], linestyle=styles[name]["linestyle"])
        ax_curve.plot(angles_deg, regressed[sample, 0, :, n].numpy(), color=REGRESSION_COLOR, linewidth=1.8,
                      linestyle="--", label=REGRESSION)
        ax_curve.set_title(f"order {n}", fontsize=10.5, color=INK)

        angle_axis(ax_error, window_edges)
        for name, error in true_error.items():
            style = ({"color": REGRESSION_COLOR, "linestyle": "--"} if name == REGRESSION else
                     {"color": styles[name]["color"], "linestyle": styles[name]["linestyle"]})
            ax_error.plot(angles_deg, error[:, n], linewidth=1.6, **style)
        ax_error.set_yscale("log")
        ax_error.set_xlabel("angle θ (deg)", fontsize=9, color=SECONDARY)
    axes[0, 0].set_ylabel(f"$a_n(\\theta)$, test sample {sample}", fontsize=9, color=SECONDARY)
    axes[1, 0].set_ylabel(f"relative RMS error over\nthe {len(dataset)} test samples", fontsize=9, color=SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
    labels.append(f"measured views (±{config.angle_max_deg:g}°)")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Moment curves of orders 0 to 4: extended from the measured window vs predicted by the network",
                 fontsize=12, color=INK)
    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    save(fig, args.out_figure, suffixes=("pdf",))
    print(f"\nSaved {args.out_figure} and {args.out_csv}")


if __name__ == "__main__":
    main()

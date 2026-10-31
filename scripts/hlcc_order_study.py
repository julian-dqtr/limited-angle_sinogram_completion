"""Why the HLCC moment regression stops at order 4, on the test split (inference only).

    python -m scripts.hlcc_order_study
    python -m scripts.hlcc_order_study --models UNet2D LinearInterp --max_order 7

The moment curve of order n has n + 1 coefficients, fitted on the 51 noisy acquired views and evaluated on the 129
missing ones (src/hlcc.py). Per order, the three panels show:
  (a) how much the regression amplifies the noise of the measured moments (geometry only);
  (b) the error of the regressed moment curves on the missing views, next to the moments of the model outputs;
  (c) the change of the wedge MSE when the orders 0..K of a model are replaced by the regression (the _P<K> models).

Outputs: outputs/figures/hlcc_order_selection.{pdf,png}, outputs/figures/hlcc_moments_by_order.{pdf,png} (panel b)
and results/evaluation/hlcc_order_study.csv.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import GeometryConfig
from src.dataset import SinogramCompletionDataset
from src.geometry import ParallelGeometry
from src.hlcc import HLCCMomentProjection, harmonic_basis
from src.models.factory import get_model
from src.paths import EVALUATION_DIR, FIGURES_DIR
from src.plotting import AXIS, BAND, FAMILY_COLORS, GRID, INK, SECONDARY, model_family, signed, style_axis

torch.backends.cudnn.enabled = False  # Tesla K80

VARIANT_STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "v")]
REGRESSION = "HLCC regression (from the 51 measured views)"


def series_styles(model_names) -> dict:
    """Colour of the family; models of the same family differ by line style and marker."""
    styles, seen = {}, {}
    for name in model_names:
        family = model_family(name)
        index = seen.setdefault(family, 0)
        seen[family] += 1
        linestyle, marker = VARIANT_STYLES[index % len(VARIANT_STYLES)]
        styles[name] = {"color": FAMILY_COLORS.get(family, SECONDARY), "linestyle": linestyle, "marker": marker}
    return styles


@torch.no_grad()
def predict(model, incomplete: torch.Tensor, device, batch_size: int = 25) -> torch.Tensor:
    return torch.cat([model(incomplete[i:i + batch_size].to(device)).float().cpu()
                      for i in range(0, len(incomplete), batch_size)])


def noise_gain(projection: HLCCMomentProjection) -> np.ndarray:
    """Variance of a regressed moment on a missing view for unit white noise on the measured moments, per order."""
    operators = projection.extrapolation[:, ~projection.acquired]  # [K+1, V_missing, V_acquired]
    return (operators**2).sum(dim=2).mean(dim=1).numpy()


def moment_error(moments: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    """Relative error of the moment curves per order, pooled over samples and views: [K+1]."""
    return ((moments - reference).pow(2).sum(dim=(0, 1, 2)) / reference.pow(2).sum(dim=(0, 1, 2))).sqrt().numpy()


def order_axis(ax, max_order: int) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    style_axis(ax)
    ax.set_xticks(range(max_order + 1))
    ax.set_xlim(-0.4, max_order + 0.4)


def mark_last_useful_order(ax, last_order: int, max_order: int, label: bool = True) -> None:
    """Grey out the orders where the regression is less accurate than the models."""
    if last_order >= max_order:
        return
    ax.axvspan(last_order + 0.5, max_order + 0.4, color=BAND, zorder=0)
    if label:
        ax.text(last_order + 0.6, 0.97, "regression less accurate\nthan the models", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8, color=SECONDARY)


def plot_moment_errors(ax, orders, regression_error, model_errors, styles, last_order) -> None:
    mark_last_useful_order(ax, last_order, orders[-1])
    marker_style = dict(markersize=6.5, markeredgecolor="white", markeredgewidth=1.2)
    for name, errors in model_errors.items():
        ax.plot(orders, errors, linewidth=1.8, label=name, **marker_style, **styles[name])
    ax.plot(orders, regression_error, color=INK, linewidth=2.0, marker="D", label=REGRESSION, **marker_style)
    ax.set_yscale("log")
    ax.set_xlabel("HLCC order n (degree of the Chebyshev polynomial along the detector)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("relative error of the moment curve $a_n(\\theta)$\non the missing views", fontsize=9,
                  color=SECONDARY)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    order_axis(ax, orders[-1])


def save(fig, path: Path, suffixes=("pdf", "png")) -> None:
    for suffix in suffixes:
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Empirical choice of the highest order of the HLCC moment regression")
    parser.add_argument("--models", nargs="+", default=["UNet2D", "UNet2dHLCC"])
    parser.add_argument("--num_samples", type=int, default=100, help="Size of the test split")
    parser.add_argument("--max_order", type=int, default=7)
    parser.add_argument("--noise_level", type=float, default=1e5)
    parser.add_argument("--figures_dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--out_csv", type=Path, default=EVALUATION_DIR / "hlcc_order_study.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GeometryConfig()
    geom = ParallelGeometry.from_config(config)
    missing = torch.from_numpy(~geom.acquired_view_mask)
    orders = list(range(args.max_order + 1))

    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level,
                                        geometry_config=config)
    samples = [dataset[i] for i in range(len(dataset))]
    incomplete = torch.stack([s[0] for s in samples])  # noisy measurements
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

    # (a) conditioning and noise amplification of the regression
    projection = HLCCMomentProjection(geom, args.max_order)
    gain = noise_gain(projection)
    condition = np.array([torch.linalg.cond(harmonic_basis(geom.angles, n)[~missing]).item() for n in orders])

    # (b) accuracy of the moment curves on the missing views
    true_moments = projection.moments(full)[:, :, missing]
    regression_error = moment_error(projection.regressed_moments(incomplete)[:, :, missing], true_moments)
    model_errors = {name: moment_error(projection.moments(pred)[:, :, missing], true_moments)
                    for name, pred in predictions.items()}
    better = regression_error < np.min(list(model_errors.values()), axis=0)
    last_order = int(np.argmin(better)) - 1 if not better.all() else args.max_order  # regression beats every model

    # (c) wedge MSE after the projection of the orders 0..K
    def wedge_mse(sinograms: torch.Tensor) -> float:
        return ((sinograms - full)[:, :, missing] ** 2).mean().item()

    mse_before = {name: wedge_mse(pred) for name, pred in predictions.items()}
    mse_change = {name: np.array([100.0 * (wedge_mse(HLCCMomentProjection(geom, K)(pred, incomplete))
                                           / mse_before[name] - 1.0) for K in orders])
                  for name, pred in predictions.items()}
    best_order = {name: int(np.argmin(change)) for name, change in mse_change.items()}

    table = pd.DataFrame({"order": orders, "condition_number": condition, "noise_gain": gain,
                          "moment_error_regression": regression_error})
    for name in predictions:
        table[f"moment_error_{name}"] = model_errors[name]
        table[f"wedge_mse_change_percent_{name}"] = mse_change[name]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"\n{len(dataset)} test samples, noise I0 = {args.noise_level:g}, "
          f"acquired window +/-{config.angle_max_deg:g} deg")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3g}"))
    print(f"\nThe regression is more accurate than every model up to order {last_order}.")
    for name in predictions:
        print(f"{name}: wedge MSE {mse_before[name]:.3e}, lowest after projecting the orders 0..{best_order[name]} "
              f"({mse_change[name][best_order[name]]:+.1f} %)")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_gain, ax_moments, ax_mse) = plt.subplots(1, 3, figsize=(16.5, 4.8), layout="constrained")

    mark_last_useful_order(ax_gain, last_order, args.max_order, label=False)
    ax_gain.axhline(1.0, color=AXIS, linewidth=1.0)
    ax_gain.text(args.max_order + 0.35, 1.0, "noise kept as it is", va="bottom", ha="right", fontsize=8,
                 color=SECONDARY)
    ax_gain.plot(orders, gain, color=INK, linewidth=2.0, marker="D", markersize=6.5, markeredgecolor="white",
                 markeredgewidth=1.2)
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel("HLCC order n", fontsize=9, color=SECONDARY)
    ax_gain.set_ylabel("variance of a regressed moment / variance of a measured one", fontsize=9, color=SECONDARY)
    ax_gain.set_title("(a) Noise amplification of the regression", fontsize=10.5, color=INK, loc="left")
    order_axis(ax_gain, args.max_order)

    plot_moment_errors(ax_moments, orders, regression_error, model_errors, styles, last_order)
    ax_moments.set_title("(b) Error of the moment curves on the missing views", fontsize=10.5, color=INK, loc="left")

    mark_last_useful_order(ax_mse, last_order, args.max_order, label=False)
    lowest = min(change.min() for change in mse_change.values())
    top = max(5.0, -0.6 * lowest)
    ax_mse.axhline(0.0, color=AXIS, linewidth=1.0)
    for name, change in mse_change.items():
        # Values above the axis are written as text: the line stops where it leaves the plot.
        drawn = np.where(change > top, np.nan, change)
        if (change > top).any():
            drawn[int(np.argmax(change > top))] = top
        ax_mse.plot(orders, drawn, linewidth=1.8, markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                    markevery=[K for K in orders if change[K] <= top],
                    label=f"{name}: lowest at K = {best_order[name]} ({signed(change[best_order[name]], 1)})",
                    **styles[name])
    off_scale = [f"K = {K}:  " + "  /  ".join(signed(change[K], 0) for change in mse_change.values())
                 for K in orders if any(change[K] > top for change in mse_change.values())]
    if off_scale:
        ax_mse.text(0.02, 0.97, "\n".join(["above the axis (" + " / ".join(mse_change) + "):", *off_scale]),
                    transform=ax_mse.transAxes, ha="left", va="top", fontsize=8, color=SECONDARY, linespacing=1.5)
    ax_mse.set_ylim(1.3 * lowest, 1.05 * top)
    ax_mse.set_xlabel("highest projected order K (orders 0..K replaced by the regression)", fontsize=9,
                      color=SECONDARY)
    ax_mse.set_ylabel("change of the wedge MSE (%)", fontsize=9, color=SECONDARY)
    ax_mse.set_title("(c) Wedge MSE after the projection (below 0: better)", fontsize=10.5, color=INK, loc="left")
    ax_mse.legend(frameon=False, fontsize=8.5, loc="lower left")
    order_axis(ax_mse, args.max_order)

    fig.suptitle(f"HLCC moment regression from a ±{config.angle_max_deg:g}° window: useful up to order {last_order} "
                 f"({len(dataset)} test samples)", fontsize=12, color=INK)
    save(fig, args.figures_dir / "hlcc_order_selection")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    plot_moment_errors(ax, orders, regression_error, model_errors, styles, last_order)
    save(fig, args.figures_dir / "hlcc_moments_by_order")
    print(f"\nSaved {args.figures_dir}/hlcc_order_selection.pdf|png, hlcc_moments_by_order.pdf|png and {args.out_csv}")


if __name__ == "__main__":
    main()

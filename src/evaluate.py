"""Evaluate the models on the test split.

    python -m src.evaluate
    python -m src.evaluate --models ZeroFilling LinearInterp UNet2D GCN_L6 SNN_L6 --num_samples 200

Outputs, in --out_dir:
    model_comparison_metrics.csv   one row per (sample, model): sinogram and image metrics, the Dirichlet-energy
                                   ratios of the wedge and the HLCC residuals hlcc_residual_<n>
    per_view_mse.csv               MSE of every view, per model
    per_view_var.csv               inter-sample variance of every view, per model and for the ground truth (the
                                   floor of a prediction that ignores the measurements)

A model whose checkpoint is missing or does not match the geometry is reported as not evaluated.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import GeometryConfig
from src.dataset import SinogramCompletionDataset
from src.geometry import ParallelGeometry, fbp_reconstruction
from src.losses import HelgasonLudwigLoss
from src.metrics import PerViewMoments, sinogram_metrics, smoothness_metrics, ssim
from src.models.factory import get_model
from src.paths import EVALUATION_DIR

torch.backends.cudnn.enabled = False  # Tesla K80

# The CSVs are read by results/model_analysis.ipynb and scripts/data_study_report.py: a run without --models
# must keep every model. Those without a checkpoint are skipped.
DEFAULT_MODELS = [
    "ZeroFilling", "LinearInterp",
    "UNet2D", "UNet2dHLCC", "UNet2D_P4", "UNet2dHLCC_P4",
    "UNet2D_N200", "UNet2dHLCC_N200", "UNet2D_N200_P4", "UNet2dHLCC_N200_P4",
    "GCN_L6", "GCN_L12", "GCN_L18", "GCN_L6_P4", "GCN_L12_P4", "GCN_L18_P4",
    "SNN_L6", "SNN_L12", "SNN_L18", "SNN_L6_P4", "SNN_L12_P4", "SNN_L18_P4",
    # seed-1 replicas of the L18 pair, to measure the run-to-run noise
    "GCN_L18_S1", "SNN_L18_S1", "GCN_L18_S1_P4", "SNN_L18_S1_P4",
    # U-Net trained through the moment regression: its prediction is _PL4_P4, _PL4 is the raw network output
    "UNet2D_PL4", "UNet2D_PL4_P4",
]
GROUND_TRUTH = "GroundTruth"
HLCC_METRIC_ORDER = 4


def hlcc_metrics(sinogram: np.ndarray, consistency: HelgasonLudwigLoss) -> dict:
    residuals = consistency.relative_residuals(torch.from_numpy(np.ascontiguousarray(sinogram)))[0]
    return {f"hlcc_residual_{order}": float(value) for order, value in enumerate(residuals)}


def image_metrics(sinogram: np.ndarray, phantom: np.ndarray, config: GeometryConfig) -> dict:
    """FBP of a normalised sinogram against the phantom (values in [0, 1])."""
    reco = fbp_reconstruction(sinogram * config.sino_norm, config)
    mse = float(np.mean((reco - phantom) ** 2))
    return {"img_psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12))), "img_ssim": ssim(phantom, reco, data_range=1.0)}


@torch.no_grad()
def evaluate_models(models: dict, dataset, device, config: GeometryConfig):
    geom = ParallelGeometry.from_config(config)
    missing_views = ~geom.acquired_view_mask
    consistency = HelgasonLudwigLoss(geom, max_order=HLCC_METRIC_ORDER)

    rows, hlcc_ground_truth = [], []
    per_view_sq_err = {name: np.zeros(geom.num_views) for name in models}
    moments = {name: PerViewMoments() for name in [GROUND_TRUTH, *models]}

    for i in tqdm(range(len(dataset)), desc="Evaluating samples"):
        incomplete, full, phantom = dataset[i]
        full_np, phantom_np = full[0].numpy(), phantom[0].numpy()
        moments[GROUND_TRUTH].update(full_np)
        hlcc_ground_truth.append(hlcc_metrics(full_np, consistency))
        if i == 0:  # Sample -1: FBP of the ground truth, the ceiling of the image metrics
            rows.append({"Sample": -1, "Model": "GroundTruthSinogram (FBP reference)",
                         **image_metrics(full_np, phantom_np, config)})

        for name, model in models.items():
            pred = model(incomplete.unsqueeze(0).to(device))[0, 0].float().cpu().numpy()
            per_view_sq_err[name] += ((pred - full_np) ** 2).mean(axis=1)
            moments[name].update(pred)
            rows.append({
                "Sample": i, "Model": name,
                **sinogram_metrics(pred, full_np, missing_views),
                **smoothness_metrics(pred, full_np, missing_views),
                **hlcc_metrics(pred, consistency),
                **image_metrics(pred, phantom_np, config),
            })

    def per_view_table(values: dict) -> pd.DataFrame:
        table = pd.DataFrame(values)
        table.insert(0, "angle_deg", np.rad2deg(geom.angles))
        table.insert(1, "acquired", geom.acquired_view_mask)
        return table

    per_view = per_view_table({name: err / len(dataset) for name, err in per_view_sq_err.items()})
    per_view_var = per_view_table({name: m.per_view_variance() for name, m in moments.items()})
    return pd.DataFrame(rows), per_view, per_view_var, pd.DataFrame(hlcc_ground_truth)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the models on the test split")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out_dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--num_samples", type=int, default=200, help="Size of the test split")
    parser.add_argument("--noise_level", type=float, default=1e5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GeometryConfig()
    dataset = SinogramCompletionDataset(args.num_samples, split="test", noise_level=args.noise_level,
                                        geometry_config=config)

    models, num_params = {}, {}
    for name in args.models:
        try:
            model, _ = get_model(name, device, geometry_config=config)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[NOT EVALUATED] {name}: {str(exc).splitlines()[0]}")
            continue
        models[name] = model
        num_params[name] = sum(p.numel() for p in model.parameters()) if isinstance(model, torch.nn.Module) else 0
    if not models:
        print("No model could be loaded: train them first with src.train.")
        return

    results, per_view, per_view_var, hlcc_ground_truth = evaluate_models(models, dataset, device, config)
    results.to_csv(args.out_dir / "model_comparison_metrics.csv", index=False)
    per_view.to_csv(args.out_dir / "per_view_mse.csv", index=False)
    per_view_var.to_csv(args.out_dir / "per_view_var.csv", index=False)

    metric_cols = ["mse_wedge", "psnr_wedge", "ssim_wedge", "ssim", "img_psnr", "img_ssim"]
    samples = results[results["Sample"] >= 0]
    per_model = samples.groupby("Model", sort=False)[metric_cols]
    mean, std = per_model.mean(), per_model.std()
    summary = pd.DataFrame({"# Params": [f"{num_params[m]:,}" for m in mean.index]}, index=mean.index)
    for col in metric_cols:
        summary[col] = [f"{mean.loc[m, col]:.4f} ± {std.loc[m, col]:.4f}" for m in mean.index]

    print(f"\nTest split: {len(dataset)} samples, sinogram data_range = 1.0, image metrics after FBP (ram-lak)")
    print(results[results["Sample"] < 0][["Model", "img_psnr", "img_ssim"]].to_string(index=False))
    print(summary.to_string())

    hlcc_summary = samples.groupby("Model", sort=False)[list(hlcc_ground_truth.columns)].mean()
    hlcc_summary.loc[GROUND_TRUTH] = hlcc_ground_truth.mean()
    print("\nHLCC residual per order, mean over the samples (0: consistent sinogram)")
    print(hlcc_summary.to_string(float_format=lambda value: f"{value:.2e}"))
    print(f"\nSaved to {args.out_dir}/model_comparison_metrics.csv, per_view_mse.csv and per_view_var.csv")


if __name__ == "__main__":
    main()

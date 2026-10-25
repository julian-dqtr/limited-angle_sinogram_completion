"""Peak GPU memory of a training step, for the runs trained before train.py recorded it.

    python -m scripts.measure_peak_vram                 # every run without peak_memory_gib
    python -m scripts.measure_peak_vram --runs GCN_L18

Two training steps (forward, loss, backward, clipping, AdamW) on one GPU with the per-GPU batch of the run, the
architecture read from its checkpoint. Written to results/tables/peak_vram.csv, read by results/model_analysis.ipynb.
Without DDP, whose gradient buckets add about one copy of the gradients (a few MB here).
"""
import argparse
import json
import sys
from datetime import date

import pandas as pd
import torch

from src.checkpoint import load_checkpoint
from src.config import GeometryConfig
from src.geometry import ParallelGeometry
from src.losses import AnnealedLoss
from src.models.factory import build_model
from src.paths import CHECKPOINT_DIR, RESULTS_DIR

torch.backends.cudnn.enabled = False  # Tesla K80
OUT_CSV = RESULTS_DIR / "tables" / "peak_vram.csv"


def peak_training_memory(model_name: str, model_config: dict, physics: bool, batch_size: int, device) -> float:
    """Peak memory (GiB) allocated by PyTorch over two training steps of a fresh model."""
    config = GeometryConfig()
    model, _ = build_model(model_name, config, **model_config)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = AnnealedLoss(ParallelGeometry.from_config(config)).to(device) if physics else None
    shape = (batch_size, 1, len(config.full_angles), config.det_col_count)
    incomplete, target = torch.rand(shape, device=device), torch.rand(shape, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        pred = model(incomplete)
        if loss_fn is not None:
            loss = loss_fn(pred[:, 0], target[:, 0], 0)[0]
        else:
            loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None, help="Checkpoint folders (default: runs without a peak)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("A GPU is needed to measure GPU memory.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    runs = args.runs or sorted(p.parent.name for p in CHECKPOINT_DIR.glob("*/training_stats.json")
                               if "peak_memory_gib" not in json.loads(p.read_text()))
    rows = []
    for run in runs:
        payload = load_checkpoint(CHECKPOINT_DIR / run / "best_model.pt", map_location="cpu")
        meta = payload.get("meta", {})
        batch_size = int(meta.get("args", {}).get("batch_size", 4))
        peak = peak_training_memory(payload["model_name"], payload["model_config"], bool(meta.get("physics", False)),
                                    batch_size, device)
        rows.append({"run": run, "model": payload["model_name"], "batch_size_per_gpu": batch_size,
                     "peak_memory_gib": round(peak, 3), "gpu": props.name,
                     "gpu_memory_gib": round(props.total_memory / 2**30, 2), "measured_on": date.today().isoformat()})
        print(f"{run:18s} batch {batch_size}: peak {peak:.2f} GiB on {props.name}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if OUT_CSV.exists():  # keep the measurements of the other runs
        previous = pd.read_csv(OUT_CSV)
        table = pd.concat([previous[~previous["run"].isin(table["run"])], table]).sort_values("run")
    table.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()

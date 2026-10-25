"""Self-describing checkpoints: architecture, geometry, weights and metadata in one file.

    {"format_version": 1, "model_name": "SNN", "model_config": {...}, "geometry": GeometryConfig.to_dict(),
     "model_state": state_dict, "meta": {"epoch": ..., "val_metrics": ..., "git_commit": ..., ...}}
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.paths import PROJECT_ROOT

FORMAT_VERSION = 1


def current_git_commit() -> Optional[str]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


def save_checkpoint(path: Path, model: torch.nn.Module, model_name: str, model_config: Dict[str, Any],
                    geometry_config, meta: Optional[Dict[str, Any]] = None) -> None:
    model = getattr(model, "module", model)  # unwrap DDP
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_config": dict(model_config),
        "geometry": geometry_config.to_dict(),
        "model_state": model.state_dict(),
        "meta": {"git_commit": current_git_commit(), **(meta or {})},
    }
    # Write then rename, so that a killed job never leaves a truncated checkpoint.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not (isinstance(payload, dict) and "model_state" in payload and "model_config" in payload):
        raise ValueError(f"{path} is not a self-describing checkpoint (bare state_dict?): retrain it with src.train.")
    return payload

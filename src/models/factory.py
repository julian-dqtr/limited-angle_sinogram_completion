"""Build, name and reload every model of the protocol.

Learned models, all called as ``completed = model(incomplete)``:

    "UNet2D"      U-Net, MSE loss
    "UNet2dHLCC"  the same U-Net, MSE + HLCC loss
    "GCN"         view-graph GCN (identity transport)
    "SNN"         SinoSheavesNN (shift transport)

Non-learned baselines: "ZeroFilling", "LinearInterp".

Suffixes, in this order, select a run or a variant without changing the architecture:

    "SNN_L12"           depth of a graph model (checkpoint folder SNN_L12)
    "UNet2D_N200"       free tags, part of the run name (train.py --run_name UNet2D_N200)
    "UNet2D_P4"         the model followed by the HLCC moment regression of orders 0..4 (src/hlcc.py), no training
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

from src.checkpoint import load_checkpoint
from src.config import GeometryConfig
from src.paths import CHECKPOINT_DIR

GRAPH_MODELS = ("GCN", "SNN")
UNET_MODELS = ("UNet2D", "UNet2dHLCC")
LEARNED_MODELS = UNET_MODELS + GRAPH_MODELS
BASELINES = ("ZeroFilling", "LinearInterp")

DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "UNet2D": {"filters": 32},
    "UNet2dHLCC": {"filters": 32},
    "GCN": {"num_stalks": 32, "num_layers": 6, "k": 12, "sigma_deg": 5.0},
    "SNN": {"num_stalks": 32, "num_layers": 6, "k": 12, "sigma_deg": 5.0},
}


class ModelName(NamedTuple):
    """A parsed model name, e.g. "UNet2dHLCC_N200_P4"."""
    base: str                         # "UNet2dHLCC"
    overrides: Dict[str, Any]         # architecture implied by the name: {"num_layers": 12} for "SNN_L12"
    tags: Tuple[str, ...]             # ("N200",), part of the run name
    projection_order: Optional[int]   # 4 for "_P4", None without moment regression


def parse_full_model_name(name: str) -> ModelName:
    match = re.fullmatch(r"(?P<base>[A-Za-z0-9]+?)(?:_L(?P<layers>\d+))?(?P<tags>(?:_[A-Za-z][A-Za-z0-9]*)*)", name)
    if match is None:
        raise ValueError(f"Cannot parse model name: {name}")
    base = match["base"]
    overrides: Dict[str, Any] = {}
    if match["layers"] is not None:
        if base not in GRAPH_MODELS:
            raise ValueError(f"The depth suffix _L<n> only applies to {GRAPH_MODELS}, got {name}")
        overrides["num_layers"] = int(match["layers"])
    tags = tuple(tag for tag in match["tags"].split("_") if tag)
    projection_order = None
    if tags and re.fullmatch(r"P\d+", tags[-1]):
        projection_order, tags = int(tags[-1][1:]), tags[:-1]
    return ModelName(base, overrides, tags, projection_order)


def parse_model_name(name: str) -> Tuple[str, Dict[str, Any]]:
    """"SNN_L12" -> ("SNN", {"num_layers": 12})."""
    parsed = parse_full_model_name(name)
    return parsed.base, parsed.overrides


def default_run_name(model_name: str, model_config: Dict[str, Any]) -> str:
    """Checkpoint folder of a run: one folder per depth for the graph models."""
    if model_name in GRAPH_MODELS:
        return f"{model_name}_L{model_config['num_layers']}"
    return model_name


def run_name_of(name: str, **model_config) -> str:
    """Checkpoint folder of a full model name: "SNN" -> "SNN_L6", "UNet2D_N200_P4" -> "UNet2D_N200"."""
    parsed = parse_full_model_name(name)
    if parsed.base in BASELINES:
        return parsed.base
    config = {**DEFAULT_MODEL_CONFIG.get(parsed.base, {}), **model_config, **parsed.overrides}
    return "_".join([default_run_name(parsed.base, config), *parsed.tags])


def build_model(model_name: str, geometry_config: Optional[GeometryConfig] = None, **model_config):
    """Fresh (untrained) learned model. Returns (model, full model config)."""
    base, overrides = parse_model_name(model_name)
    if base not in LEARNED_MODELS:
        raise ValueError(f"Unknown learned model: {model_name}. Available: {LEARNED_MODELS}")
    for key, value in overrides.items():
        if key in model_config and model_config[key] != value:
            raise ValueError(f"'{model_name}' implies {key}={value} but {key}={model_config[key]} was requested.")
    config = {**DEFAULT_MODEL_CONFIG[base], **model_config, **overrides}
    geometry_config = geometry_config or GeometryConfig()

    if base == "UNet2D":
        from src.models.unet import SinogramUNet as cls
    elif base == "UNet2dHLCC":
        from src.models.unet import SinogramUNetHLCC as cls
    elif base == "GCN":
        from src.models.view_graph.model import SinoGCN as cls
    else:
        from src.models.view_graph.model import SinoSheafNet as cls
    return cls(geometry_config=geometry_config, **config), config


def get_model(model_name: str, device: torch.device, load_weights: bool = True,
              checkpoint_path: Optional[Path] = None, geometry_config: Optional[GeometryConfig] = None, **kwargs):
    """Model ready for inference, with its trained weights by default. Returns (model, is_nn).

    The architecture and the geometry are read from the checkpoint and the load is strict: a missing checkpoint,
    another model type or another geometry raises, so that an evaluation never scores a random network.
    """
    parsed = parse_full_model_name(model_name)
    base, overrides = parsed.base, parsed.overrides
    geometry_config = geometry_config or GeometryConfig()

    if parsed.projection_order is not None:
        from src.geometry import ParallelGeometry
        from src.hlcc import HLCCProjectedModel

        inner_name = model_name[: model_name.rfind("_P")]
        inner, is_nn = get_model(inner_name, device, load_weights, checkpoint_path, geometry_config, **kwargs)
        geom = ParallelGeometry.from_config(geometry_config)
        return HLCCProjectedModel(inner, geom, parsed.projection_order).to(device).eval(), is_nn

    if base == "ZeroFilling":
        from src.models.baselines import zero_filling
        return zero_filling, False
    if base == "LinearInterp":
        from src.models.baselines import linear_interpolation
        return linear_interpolation, False

    if not load_weights:
        model, _ = build_model(model_name, geometry_config, **kwargs)
        return model.to(device).eval(), True

    checkpoint_path = Path(checkpoint_path or CHECKPOINT_DIR / run_name_of(model_name, **kwargs) / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint for {model_name} at {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path, map_location=device)
    if payload["model_name"] != base:
        raise ValueError(f"{checkpoint_path} contains a {payload['model_name']} model, expected {base}")
    for key, value in overrides.items():
        if payload["model_config"].get(key) != value:
            raise ValueError(f"'{model_name}' implies {key}={value} but {checkpoint_path} has "
                             f"{key}={payload['model_config'].get(key)}")
    checkpoint_geometry = GeometryConfig.from_dict(payload["geometry"])
    if checkpoint_geometry != geometry_config:
        raise ValueError(f"{checkpoint_path} was trained on a different geometry:\n"
                         f"  checkpoint: {checkpoint_geometry}\n  requested:  {geometry_config}")

    model, _ = build_model(base, checkpoint_geometry, **payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    print(f"[+] Loaded {model_name} from {checkpoint_path} (config: {payload['model_config']})")
    return model.to(device).eval(), True

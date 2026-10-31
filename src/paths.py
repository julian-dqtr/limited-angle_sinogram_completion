"""Locations of the generated files (all ignored by git, except results/)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "2d" / "checkpoints"
VISUALISATION_DIR = OUTPUT_DIR / "2d" / "visualisation"
OPTUNA_DIR = OUTPUT_DIR / "2d" / "optuna"
LOG_DIR = OUTPUT_DIR / "2d" / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OPTUNA_DB_DIR = PROJECT_ROOT / "db"

# Thesis tables and figures, versioned, with the per-sample evaluation they are computed from.
RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"

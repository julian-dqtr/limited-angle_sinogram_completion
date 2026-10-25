"""Optuna search on top of the training loop of src/train.py.

    python -m src.optuna_search --model UNet2dHLCC --n-trials 50
    python -m src.optuna_search --model UNet2dHLCC --physics-only --n-trials 50
    python -m src.optuna_search --model GCN --num_layers 12 --optimizer-only --n-trials 4

One worker per GPU: create the study with --create-only, then start several workers on it with
CUDA_VISIBLE_DEVICES=<i>.

- The objective is the validation wedge MSE of the best epoch, the metric that selects checkpoints.
- The ablation variables of the graph models (num_layers, k, graph_type) are fixed by the CLI, never searched.
- --physics-only searches the HLCC terms on top of the default U-Net; --optimizer-only searches lr and
  weight_decay only, so that GCN and SNN keep the same graph and parameter count.
- Each trial records its model configuration ("model_config" user attribute) for scripts/launch_best_training.py.
- Trials never write checkpoints, and best_params.json is rewritten after every completed trial.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import optuna

from src.models.factory import GRAPH_MODELS, LEARNED_MODELS, default_run_name
from src.paths import OPTUNA_DB_DIR, OPTUNA_DIR
from src.train import SELECTION_METRIC, model_config_from_args, run_training, uses_physics
from src.train import build_parser as build_train_parser


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--model", type=str, required=True, choices=LEARNED_MODELS)
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials of this worker")
    parser.add_argument("--n-epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--n-samples", type=int, default=500, help="Training samples per trial")
    parser.add_argument("--n-val", type=int, default=100, help="Validation samples per trial")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physics", type=str, default="auto", choices=["auto", "none", "hlcc"])
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--physics-only", action="store_true", help="Only search the HLCC terms")
    search.add_argument("--optimizer-only", action="store_true", help="Only search lr and weight_decay")
    # With the HLCC terms ~1 and the MSE ~2e-2 on zero filling, lambda = 1e-2 already gives the physics term
    # about half the weight of the data term.
    parser.add_argument("--lambda-range", type=float, nargs=2, default=(1e-5, 1e-2), metavar=("LOW", "HIGH"),
                        help="Log-uniform range of lambda_m0, lambda_m1 and lambda_high")
    parser.add_argument("--hlcc-max-order", type=int, default=1, help="Highest HLCC order of the loss (fixed)")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--graph_type", type=str, default="knn", choices=["knn", "full"])
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--storage", type=str, default=None, help="Defaults to sqlite:///db/optuna_<study>.db")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Defaults to <run_name>_parallel, plus _physics_only / _optimizer_only")
    parser.add_argument("--create-only", action="store_true",
                        help="Create the study and exit: run it once before starting parallel workers")
    return parser.parse_args()


def make_train_args(args, trial: optuna.Trial):
    """argparse namespace of src/train.py for one trial."""
    argv = [
        "--model", args.model, "--epochs", str(args.n_epochs), "--n_samples", str(args.n_samples),
        "--n_val", str(args.n_val), "--batch_size", str(args.batch_size), "--physics", args.physics,
        "--num_layers", str(args.num_layers), "--k", str(args.k), "--graph_type", args.graph_type,
        "--hlcc_max_order", str(args.hlcc_max_order),
    ]
    train_args = build_train_parser().parse_args(argv)
    train_args.grad_checkpoint = args.grad_checkpoint
    train_args.use_wandb = args.use_wandb
    train_args.wandb_project = "dbt-sinogram-optuna-2d"

    physics = uses_physics(train_args)
    if args.physics_only and not physics:
        raise ValueError(f"--physics-only needs the HLCC loss, off for --model {args.model} --physics {args.physics}.")

    if not args.physics_only:
        train_args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        train_args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    if not (args.physics_only or args.optimizer_only):
        if args.model in GRAPH_MODELS:
            train_args.num_stalks = trial.suggest_categorical("num_stalks", [16, 32, 64])
            train_args.sigma_deg = trial.suggest_categorical("sigma_deg", [2.0, 3.5, 5.0, 10.0])
        else:
            train_args.filters = trial.suggest_categorical("filters", [16, 32, 64])
    if physics and not args.optimizer_only:
        low, high = args.lambda_range
        train_args.lambda_m0 = trial.suggest_float("lambda_m0", low, high, log=True)
        train_args.lambda_m1 = trial.suggest_float("lambda_m1", low, high, log=True)
        if args.hlcc_max_order >= 2:
            train_args.lambda_high = trial.suggest_float("lambda_high", low, high, log=True)
        train_args.anneal_epochs = trial.suggest_int("anneal_epochs", 2, max(3, args.n_epochs // 2))

    train_args.run_name = f"{args.study_name}_trial{trial.number}"
    return train_args


def ensure_sqlite_wal(storage_url: str) -> None:
    """WAL mode avoids locking errors when several workers share one SQLite study."""
    if storage_url.startswith("sqlite:///"):
        db_path = Path(storage_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        conn.commit()
        conn.close()


def main():
    args = parse_args()
    base_config = model_config_from_args(build_train_parser().parse_args(
        ["--model", args.model, "--num_layers", str(args.num_layers), "--k", str(args.k)]))
    run_name = default_run_name(args.model, base_config)
    mode_suffix = "_physics_only" if args.physics_only else "_optimizer_only" if args.optimizer_only else ""
    order_suffix = f"_K{args.hlcc_max_order}" if args.hlcc_max_order != 1 else ""
    args.study_name = args.study_name or f"{run_name}_parallel{mode_suffix}{order_suffix}"
    args.storage = args.storage or f"sqlite:///{OPTUNA_DB_DIR / f'optuna_{args.study_name}.db'}"
    best_params_path = OPTUNA_DIR / args.study_name / "best_params.json"

    ensure_sqlite_wal(args.storage)
    storage = optuna.storages.RDBStorage(url=args.storage, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(direction="minimize", study_name=args.study_name, storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5))
    if args.create_only:
        print(f"Optuna study '{args.study_name}' ready ({args.storage})")
        return

    def objective(trial: optuna.Trial) -> float:
        train_args = make_train_args(args, trial)
        trial.set_user_attr("model_config", model_config_from_args(train_args))
        best = run_training(train_args, trial=trial, save=False)
        for key, value in best.items():
            trial.set_user_attr(key, value)
        return best[SELECTION_METRIC]

    def write_best_params(study: optuna.Study, _trial) -> None:
        try:
            best = study.best_trial
        except ValueError:  # no completed trial yet
            return
        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        best_params_path.write_text(json.dumps({
            "model": args.model, "study_name": args.study_name, f"best_{SELECTION_METRIC}": best.value,
            "trial_number": best.number, "user_attrs": best.user_attrs, "params": best.params,
            "fixed": {"num_layers": args.num_layers, "k": args.k, "graph_type": args.graph_type,
                      "physics": args.physics, "batch_size": args.batch_size, "hlcc_max_order": args.hlcc_max_order,
                      "physics_only": args.physics_only, "optimizer_only": args.optimizer_only,
                      "lambda_range": list(args.lambda_range)},
        }, indent=2))

    print(f"Optuna study '{args.study_name}' ({args.storage})")
    study.optimize(objective, n_trials=args.n_trials, callbacks=[write_best_params])
    print(f"Best {SELECTION_METRIC}: {study.best_value:.6f} with {study.best_params}\nSaved to {best_params_path}")


if __name__ == "__main__":
    main()

"""Print (or start with --run) the training command of the best trial of an Optuna study.

    python -m scripts.launch_best_training SNN_L12_parallel
    python -m scripts.launch_best_training GCN_L6_parallel_optimizer_only --run
    python -m scripts.launch_best_training GCN_L12_parallel_optimizer_only --model SNN --run   # SNN, same config
    python -m scripts.launch_best_training GCN_L18_parallel_optimizer_only --seed 1 --run      # replica GCN_L18_S1

Every searched and fixed parameter is forwarded, including the architecture recorded by the best trial, so the
final run is the configuration Optuna evaluated. The thesis runs used 8 GPUs x batch 4: with fewer GPUs, keep the
effective batch of 32 with --grad_accum_steps (e.g. --nproc 1 --grad_accum_steps 8).
"""
import argparse
import json
import shlex
import subprocess

from src.paths import OPTUNA_DIR, PROJECT_ROOT

GRAPH_MODELS = ("GCN", "SNN")  # as in src/models/factory.py, not imported to avoid loading torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--nproc", type=int, default=8, help="GPUs for torchrun (1: plain python)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Batches per optimizer step")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Multiply the searched lr")
    parser.add_argument("--run", action="store_true", help="Start the training instead of only printing it")
    parser.add_argument("--seed", type=int, default=0,
                        help="Another seed (initialisation and batch order, same phantoms): run saved as <run>_S<seed>")
    parser.add_argument("--model", choices=GRAPH_MODELS, default=None,
                        help="Train the other graph model with this study's configuration")
    args = parser.parse_args()

    best = json.loads((OPTUNA_DIR / args.study_name / "best_params.json").read_text())
    params, fixed = dict(best["params"]), best["fixed"]
    model = args.model or best["model"]
    if args.model and best["model"] not in GRAPH_MODELS:
        parser.error(f"--model swaps graph models only, but '{args.study_name}' is a {best['model']} study.")
    if "lr" in params:
        params["lr"] = params["lr"] * args.lr_scale
    elif args.lr_scale != 1.0:
        parser.error("--lr-scale needs a searched lr, but this study kept the default lr (--physics-only).")

    launcher = [".venv/bin/torchrun", f"--nproc_per_node={args.nproc}"] if args.nproc > 1 else [".venv/bin/python"]
    cmd = launcher + ["-m", "src.train", "--model", model, "--epochs", str(args.epochs),
                      "--n_samples", str(args.n_samples), "--n_val", str(args.n_val),
                      "--batch_size", str(fixed["batch_size"]), "--physics", fixed["physics"], "--use-wandb"]
    if args.grad_accum_steps != 1:
        cmd += ["--grad_accum_steps", str(args.grad_accum_steps)]
    if model in GRAPH_MODELS:
        cmd += ["--num_layers", str(fixed["num_layers"]), "--k", str(fixed["k"]), "--graph_type", fixed["graph_type"]]
    if args.seed != 0:  # run name as in factory.run_name_of("GCN_L18_S1")
        run_name = f"{model}_L{fixed['num_layers']}" if model in GRAPH_MODELS else model
        cmd += ["--seed", str(args.seed), "--run_name", f"{run_name}_S{args.seed}"]
    if fixed.get("hlcc_max_order", 1) != 1:  # older studies only had the orders 0 and 1
        cmd += ["--hlcc_max_order", str(fixed["hlcc_max_order"])]
    for key, value in params.items():
        cmd += [f"--{key}", str(value)]
    # Architecture of the best trial, so that a later change of a train.py default cannot change the model.
    forwarded = set(params) | {"num_layers", "k", "graph_type"}
    for key, value in best["user_attrs"].get("model_config", {}).items():
        if key in forwarded:
            continue
        if isinstance(value, bool):
            cmd += [f"--{key}"] if value else []
        else:
            cmd += [f"--{key}", str(value)]

    print(f"Best trial #{best['trial_number']} of '{args.study_name}': {best}")
    if model != best["model"]:
        print(f"Model swapped: {best['model']} -> {model} (same configuration)")
    print("\n" + " ".join(shlex.quote(c) for c in cmd) + "\n")
    if args.run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

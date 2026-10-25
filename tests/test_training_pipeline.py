"""End-to-end smoke test: one tiny epoch, checkpoint, reload through the factory."""
import json

import pytest
import torch

from src.models.factory import build_model, get_model
from src.train import build_parser, run_training


@pytest.mark.parametrize("model_args", [
    ["--model", "UNet2D", "--filters", "8"],
    ["--model", "UNet2dHLCC", "--filters", "8"],
    ["--model", "GCN", "--num_layers", "2", "--num_stalks", "4"],
    ["--model", "SNN", "--num_layers", "2", "--num_stalks", "4", "--physics", "hlcc"],
    ["--model", "UNet2dHLCC", "--filters", "8", "--hlcc_max_order", "3", "--train_repeats", "2"],
    ["--model", "UNet2D", "--filters", "8", "--projection_order", "4"],
    ["--model", "GCN", "--num_layers", "2", "--num_stalks", "4", "--grad_accum_steps", "2"],
])
def test_train_save_reload(model_args, tmp_path, device):
    if device.type != "cuda":
        pytest.skip("the ASTRA projector needs CUDA")
    args = build_parser().parse_args(model_args + [
        "--epochs", "1", "--n_samples", "4", "--n_val", "4", "--batch_size", "2",
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--figures-dir", str(tmp_path / "fig"),
    ])
    best = run_training(args)
    assert best["best_epoch"] == 1 and all(map(lambda v: v == v, best.values()))  # no NaN

    for artefact in ("best_model.pt", "training_stats.json", "history.json"):
        assert (tmp_path / "ckpt" / artefact).exists()
    logged = json.loads((tmp_path / "ckpt" / "history.json").read_text())[0]  # one HLCC term per order
    expected = {f"train/hlcc_m{n}" for n in range(args.hlcc_max_order + 1)}
    assert {key for key in logged if key.startswith("train/hlcc_m")} == expected
    model, is_nn = get_model(args.model, device, checkpoint_path=tmp_path / "ckpt" / "best_model.pt")
    assert is_nn
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 180, 128, device=device))
    assert out.shape == (1, 1, 180, 128)


@pytest.mark.parametrize("name,config_overrides",
                         [("UNet2D", {"filters": 8}), ("SNN", {"num_layers": 2, "num_stalks": 4})])
def test_gradient_accumulation_matches_the_full_batch(name, config_overrides, config, clean_val_batch):
    """No normalisation depends on the batch: accumulating two half batches gives the full-batch gradient, so
    1 GPU with --grad_accum_steps 8 optimises the same objective as 8 GPUs."""
    torch.manual_seed(0)
    incomplete, full = clean_val_batch[0][:4], clean_val_batch[1][:4]
    model, _ = build_model(name, config, **config_overrides)

    def gradients(chunks):
        model.zero_grad()
        for x, y in chunks:
            (torch.nn.functional.mse_loss(model(x), y) / len(chunks)).backward()
        return [p.grad.clone() for p in model.parameters()]

    full_batch = gradients([(incomplete, full)])
    accumulated = gradients([(incomplete[:2], full[:2]), (incomplete[2:], full[2:])])
    for a, b in zip(full_batch, accumulated):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)

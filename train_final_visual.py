"""Train the final all-sequence chess visual model using OOF-selected budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_builder import (
    VisualCandidateDataset, apply_standardizer, fit_feature_standardizer,
    load_arrays, sequence_balanced_weights,
)
from losses import visual_fusion_loss
from visual_models import make_visual_model, parameter_count


def _sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = root / cfg["paths"]["artifacts"]
    arrays = load_arrays(artifacts / "candidate_dataset.npz")
    visual_file = (
        "visual_responses_dynamic.npz" if args.dynamic else "visual_responses.npz"
    )
    manifest_file = (
        "visual_run_manifest_dynamic.json" if args.dynamic
        else "visual_run_manifest.json"
    )
    visual = np.load(artifacts / visual_file, allow_pickle=False)
    oof = json.loads((artifacts / manifest_file).read_text(encoding="utf-8"))
    chess_meta = [
        x["model"] for x in oof["training"]
        if x["split"]["target_domain"] == "chess"
    ]
    epochs = max(1, int(round(np.median([x["best_epoch"] + 1 for x in chess_meta]))))
    alpha = float(np.median([x["residual_alpha"] for x in chess_meta]))
    seed = int(cfg["seed"]) + 30000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    indices = np.arange(len(arrays["gt"]))
    mean, std = fit_feature_standardizer(arrays["features"])
    scaled = dict(arrays)
    scaled["features"] = apply_standardizer(arrays["features"], mean, std)
    dataset = VisualCandidateDataset(
        scaled, visual["maps"], visual["valid"], indices
    )
    weights = sequence_balanced_weights(arrays, indices)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(indices), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    tc = cfg["visual_training"]
    loader = DataLoader(
        dataset, batch_size=int(tc["batch_size"]), sampler=sampler, num_workers=0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_channels = int(visual["maps"].shape[2])
    model = make_visual_model(
        scaled["features"].shape[-1], cfg, visual_channels=visual_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tc["learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
    )
    history = []
    for epoch in range(epochs):
        model.train()
        values = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            boxes, gt = batch["boxes"].to(device), batch["gt"].to(device)
            details = model.forward_details(
                batch["features"].to(device), boxes, batch["maps"].to(device)
            )
            loss, _ = visual_fusion_loss(
                details, boxes, gt,
                residual_limit=float(cfg["models"]["visual_residual_limit"]),
                quality_weight=float(tc["quality_weight"]),
                residual_weight=float(tc["residual_weight"]),
                temporal_weight=float(tc["temporal_weight"]),
                domain_trackves=batch["features"][:, 0, 20].to(device),
                trackves_size_weight=float(tc["trackves_size_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": float(np.mean(values))})
    output = artifacts / "deployment" / (
        "hybrid_dynamic_chess.pt" if args.dynamic else "hybrid_visual_chess.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": "visual_quality_residual",
        "state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std,
        "feature_dim": int(scaled["features"].shape[-1]), "seed": seed,
        "residual_alpha": alpha, "trained_domains": ["chess", "trackves"],
        "visual_channels": visual_channels,
        "deployment_policy": "visual_on_chess_nano_on_trackves",
    }, output)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(output.relative_to(root)),
        "checkpoint_sha256": _sha256(output), "parameters": parameter_count(model),
        "epochs": epochs, "residual_alpha": alpha, "seed": seed,
        "visual_channels": visual_channels,
        "budget_source": "Median chess-target OOF best epoch and alpha.",
        "training_samples": len(indices), "sequence_balanced_sampler": True,
        "history": history,
    }
    manifest_name = (
        "dynamic_deployment_manifest.json" if args.dynamic
        else "deployment_manifest.json"
    )
    (output.parent / manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()

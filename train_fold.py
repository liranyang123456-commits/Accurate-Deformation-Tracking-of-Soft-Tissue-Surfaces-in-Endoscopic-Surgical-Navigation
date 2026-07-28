"""Train one grouped OOF fold with train-only feature normalization."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_builder import CandidateDataset, Fold, apply_standardizer, fit_feature_standardizer
from losses import aligned_iou, fusion_loss
from models import make_model, parameter_count


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numpy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:])
    inter = np.maximum(rb - lt, 0.0).prod(axis=1)
    area_a = np.maximum(a[:, 2:], 0.0).prod(axis=1)
    area_b = np.maximum(b[:, 2:], 0.0).prod(axis=1)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


@torch.no_grad()
def _mean_iou(model, loader, device: torch.device) -> float:
    model.eval()
    values = []
    for batch in loader:
        features = batch["features"].to(device)
        boxes = batch["boxes"].to(device)
        gt = batch["gt"].to(device)
        pred, _ = model(features, boxes)
        values.append(aligned_iou(pred, gt).cpu())
    return float(torch.cat(values).mean()) if values else float("-inf")


@torch.no_grad()
def predict(
    model, arrays: dict[str, np.ndarray], indices: np.ndarray,
    device: torch.device, batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(CandidateDataset(arrays, indices), batch_size=batch_size, shuffle=False)
    model.eval()
    pred_parts, weight_parts = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        pred, weights = model(
            batch["features"].to(device), batch["boxes"].to(device)
        )
        pred_parts.append(pred.cpu().numpy())
        weight_parts.append(weights.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - start) / max(len(indices), 1)
    return np.concatenate(pred_parts), np.concatenate(weight_parts), elapsed_ms


def train_one(
    *,
    model_name: str,
    fold: Fold,
    raw_arrays: dict[str, np.ndarray],
    cfg: dict,
    checkpoint_path: Path,
    seed_offset: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    seed = int(cfg["seed"]) + seed_offset
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = fit_feature_standardizer(raw_arrays["features"][fold.train_idx])
    arrays = dict(raw_arrays)
    arrays["features"] = apply_standardizer(raw_arrays["features"], mean, std)
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        CandidateDataset(arrays, fold.train_idx),
        batch_size=int(train_cfg["batch_size"]), shuffle=True,
        num_workers=int(train_cfg["num_workers"]), drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        CandidateDataset(arrays, fold.val_idx),
        batch_size=1024, shuffle=False, num_workers=0,
    )
    model = make_model(model_name, arrays["features"].shape[-1], cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    best_iou, best_epoch, stale, best_state = -1.0, -1, 0, None
    history = []
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features, boxes, gt = (
                batch["features"].to(device), batch["boxes"].to(device),
                batch["gt"].to(device),
            )
            pred, weights = model(features, boxes)
            loss, _ = fusion_loss(
                pred, weights, boxes, gt,
                float(train_cfg["oracle_ce_weight"]),
                float(train_cfg["temporal_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_iou = _mean_iou(model, val_loader, device)
        history.append({"epoch": epoch, "loss": np.mean(losses), "val_iou": val_iou})
        if val_iou > best_iou + 1e-5:
            best_iou, best_epoch, stale = val_iou, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(train_cfg["patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_pred, _, _ = predict(model, arrays, fold.val_idx, device)
    val_base = arrays["boxes"][fold.val_idx, 0]
    val_gt = arrays["gt"][fold.val_idx]
    # Validation-only conservative residual calibration. Alpha=0 (the frozen
    # full fusion) is always available, so a learned gate is used only where
    # its held-out validation score justifies deviation.
    alpha_scores = []
    val_pairs = list(zip(
        map(str, arrays["domain"][fold.val_idx]),
        map(str, arrays["sequence"][fold.val_idx]),
    ))
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = val_base + float(alpha) * (val_pred - val_base)
        ious = _numpy_iou(blended, val_gt)
        pair_means = [
            ious[np.asarray([pair == target for pair in val_pairs])].mean()
            for target in sorted(set(val_pairs))
        ]
        alpha_scores.append((float(np.mean(pair_means)), float(alpha)))
    best_alpha_score, best_alpha = max(alpha_scores, key=lambda x: (x[0], -x[1]))
    raw_pred, weights, ms_per_frame = predict(model, arrays, fold.test_idx, device)
    test_base = arrays["boxes"][fold.test_idx, 0]
    pred = test_base + best_alpha * (raw_pred - test_base)
    identity = np.zeros_like(weights)
    identity[:, 0] = 1.0
    weights = identity + best_alpha * (weights - identity)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name, "state_dict": best_state,
            "feature_mean": mean, "feature_std": std, "seed": seed,
            "residual_alpha": best_alpha,
            "fold_id": fold.fold_id, "test_sequence": fold.test_sequence,
            "val_sequences": fold.val_sequences,
            "feature_dim": int(arrays["features"].shape[-1]),
        },
        checkpoint_path,
    )
    meta = {
        "model": model_name, "fold_id": fold.fold_id,
        "best_epoch": best_epoch, "best_val_iou": best_iou,
        "epochs_ran": len(history), "parameters": parameter_count(model),
        "device": str(device), "ms_per_frame": ms_per_frame,
        "residual_alpha": best_alpha,
        "residual_alpha_val_iou": best_alpha_score,
        "history": history,
    }
    return meta, pred, weights

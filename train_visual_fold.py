"""Train one sequence-balanced visual quality/residual OOF fold."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_builder import (
    Fold, VisualCandidateDataset, apply_standardizer, fit_feature_standardizer,
    sequence_balanced_weights,
)
from losses import aligned_iou, visual_fusion_loss
from visual_models import make_visual_model, parameter_count


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _np_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:])
    inter = np.maximum(rb - lt, 0).prod(1)
    aa = np.maximum(a[:, 2:], 0).prod(1)
    ab = np.maximum(b[:, 2:], 0).prod(1)
    return inter / np.maximum(aa + ab - inter, 1e-6)


@torch.no_grad()
def predict_visual(
    model, arrays: dict[str, np.ndarray], maps: np.ndarray, valid: np.ndarray,
    indices: np.ndarray, device: torch.device, batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loader = DataLoader(
        VisualCandidateDataset(arrays, maps, valid, indices),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.eval()
    pred_parts, weight_parts, quality_parts = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in loader:
        details = model.forward_details(
            batch["features"].to(device), batch["boxes"].to(device),
            batch["maps"].to(device),
        )
        pred_parts.append(details["pred"].cpu().numpy())
        weight_parts.append(details["weights"].cpu().numpy())
        quality_parts.append(details["quality"].cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - start) / max(len(indices), 1)
    return (
        np.concatenate(pred_parts), np.concatenate(weight_parts),
        np.concatenate(quality_parts), ms,
    )


def _equal_sequence_score(
    pred: np.ndarray, gt: np.ndarray, domains: np.ndarray, sequences: np.ndarray
) -> float:
    iou = _np_iou(pred, gt)
    pairs = list(zip(map(str, domains), map(str, sequences)))
    return float(np.mean([
        iou[np.asarray([x == pair for x in pairs])].mean()
        for pair in sorted(set(pairs))
    ]))


def train_visual_one(
    *,
    fold: Fold,
    raw_arrays: dict[str, np.ndarray],
    visual_maps: np.ndarray,
    visual_valid: np.ndarray,
    cfg: dict,
    checkpoint_path: Path,
    seed_offset: int,
    target_only: bool = False,
    model_label: str = "visual_quality_residual",
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    seed = int(cfg["seed"]) + 10000 + seed_offset
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices = fold.train_idx
    if target_only:
        train_indices = train_indices[
            raw_arrays["domain"][train_indices] == fold.target_domain
        ]
    mean, std = fit_feature_standardizer(raw_arrays["features"][train_indices])
    arrays = dict(raw_arrays)
    arrays["features"] = apply_standardizer(raw_arrays["features"], mean, std)
    train_cfg = cfg["visual_training"]
    train_set = VisualCandidateDataset(
        arrays, visual_maps, visual_valid, train_indices
    )
    sample_weights = sequence_balanced_weights(raw_arrays, train_indices)
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), num_samples=len(train_indices),
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=int(train_cfg["batch_size"]), sampler=sampler,
        num_workers=int(train_cfg["num_workers"]), drop_last=False,
    )
    visual_channels = int(visual_maps.shape[2])
    model = make_visual_model(
        arrays["features"].shape[-1], cfg, visual_channels=visual_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    best_score, best_epoch, stale, best_state = -1.0, -1, 0, None
    history = []
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch["features"].to(device)
            boxes = batch["boxes"].to(device)
            gt = batch["gt"].to(device)
            details = model.forward_details(features, boxes, batch["maps"].to(device))
            loss, _ = visual_fusion_loss(
                details, boxes, gt,
                residual_limit=float(cfg["models"]["visual_residual_limit"]),
                quality_weight=float(train_cfg["quality_weight"]),
                residual_weight=float(train_cfg["residual_weight"]),
                temporal_weight=float(train_cfg["temporal_weight"]),
                domain_trackves=features[:, 0, 20],
                trackves_size_weight=float(train_cfg["trackves_size_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_pred, _, _, _ = predict_visual(
            model, arrays, visual_maps, visual_valid, fold.val_idx, device
        )
        target_mask = arrays["domain"][fold.val_idx] == fold.target_domain
        val_score = _equal_sequence_score(
            val_pred[target_mask], arrays["gt"][fold.val_idx][target_mask],
            arrays["domain"][fold.val_idx][target_mask],
            arrays["sequence"][fold.val_idx][target_mask],
        )
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "val_iou": val_score
        })
        if val_score > best_score + 1e-5:
            best_score, best_epoch, stale = val_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= int(train_cfg["patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_pred, _, _, _ = predict_visual(
        model, arrays, visual_maps, visual_valid, fold.val_idx, device
    )
    val_base = arrays["boxes"][fold.val_idx, 0]
    target_mask = arrays["domain"][fold.val_idx] == fold.target_domain
    alpha_scores = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = val_base + float(alpha) * (val_pred - val_base)
        score = _equal_sequence_score(
            blended[target_mask], arrays["gt"][fold.val_idx][target_mask],
            arrays["domain"][fold.val_idx][target_mask],
            arrays["sequence"][fold.val_idx][target_mask],
        )
        alpha_scores.append((score, float(alpha)))
    best_alpha_score, best_alpha = max(alpha_scores, key=lambda x: (x[0], -x[1]))
    raw_pred, weights, quality, ms = predict_visual(
        model, arrays, visual_maps, visual_valid, fold.test_idx, device
    )
    base = arrays["boxes"][fold.test_idx, 0]
    pred = base + best_alpha * (raw_pred - base)
    identity = np.zeros_like(weights)
    identity[:, 0] = 1.0
    weights = identity + best_alpha * (weights - identity)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": "visual_quality_residual",
        "state_dict": best_state, "feature_mean": mean, "feature_std": std,
        "feature_dim": int(arrays["features"].shape[-1]), "seed": seed,
        "fold_id": fold.fold_id, "test_sequence": fold.test_sequence,
        "val_sequences": fold.val_sequences, "residual_alpha": best_alpha,
        "visual_channels": visual_channels,
    }, checkpoint_path)
    meta = {
        "model": model_label, "fold_id": fold.fold_id,
        "best_epoch": best_epoch, "best_val_iou": best_score,
        "epochs_ran": len(history), "parameters": parameter_count(model),
        "device": str(device), "ms_per_frame": ms,
        "residual_alpha": best_alpha,
        "residual_alpha_val_iou": best_alpha_score,
        "sequence_balanced_sampler": True, "target_only_training": target_only,
        "training_samples": len(train_indices), "history": history,
    }
    return meta, pred, weights, quality

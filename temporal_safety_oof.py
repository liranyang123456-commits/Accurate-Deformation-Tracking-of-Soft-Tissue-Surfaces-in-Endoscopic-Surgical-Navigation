"""OOF temporal safety gate and TrackVes-aware center/size calibration.

The second-stage GRU is trained only on first-stage OOF predictions from
non-test sequences. It cannot see predictions made by a model trained on the
same sequence, preventing stacked-model leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from candidate_exporter import box_iou
from dataset_builder import build_dual_oof_folds, load_arrays, sequence_balanced_weights
from run_nested_oof import _prediction_rows


HISTORY = 5


def _np_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:])
    inter = np.maximum(rb - lt, 0).prod(1)
    aa = np.maximum(a[:, 2:], 0).prod(1)
    ab = np.maximum(b[:, 2:], 0).prod(1)
    return inter / np.maximum(aa + ab - inter, 1e-6)


class TemporalSafetySizeNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden, 1)
        self.correction = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 4), nn.Tanh())

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, state = self.gru(history)
        state = self.norm(state[-1])
        return self.gate(state).squeeze(-1), self.correction(state)


def _current_vectors(
    arrays: dict[str, np.ndarray], visual: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality = np.stack(visual.predicted_candidate_quality.map(json.loads)).astype(np.float32)
    weights = np.stack(visual.weights.map(json.loads)).astype(np.float32)
    pred = visual[["pred_x", "pred_y", "pred_w", "pred_h"]].to_numpy(np.float32)
    base = arrays["boxes"][:, 0].astype(np.float32)
    scale = np.stack([base[:, 2], base[:, 3], base[:, 2], base[:, 3]], axis=1)
    delta = (pred - base) / np.maximum(scale, 1.0)
    selected = arrays["features"][:, :, 8:20]
    statistics = np.concatenate([selected.mean(1), selected.std(1)], axis=1)
    current = np.concatenate([quality, weights, delta, statistics], axis=1).astype(np.float32)
    return current, pred, quality


def _history_windows(
    current: np.ndarray, arrays: dict[str, np.ndarray]
) -> np.ndarray:
    output = np.empty((len(current), HISTORY, current.shape[1]), dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for i, (domain, unit) in enumerate(zip(arrays["domain"], arrays["unit"])):
        groups.setdefault(f"{domain}:{unit}", []).append(i)
    for indices in groups.values():
        for position, index in enumerate(indices):
            available = indices[max(0, position - HISTORY + 1): position + 1]
            pad = [available[0]] * (HISTORY - len(available))
            output[index] = current[pad + available]
    return output


def _targets(
    visual_pred: np.ndarray, base: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    visual_iou = _np_iou(visual_pred, gt)
    base_iou = _np_iou(base, gt)
    gate = (visual_iou > base_iou).astype(np.float32)
    previous_scale = np.stack(
        [base[:, 2], base[:, 3], base[:, 2], base[:, 3]], axis=1
    )
    correction = np.stack([
        (gt[:, 0] - visual_pred[:, 0]) / np.maximum(previous_scale[:, 0], 1),
        (gt[:, 1] - visual_pred[:, 1]) / np.maximum(previous_scale[:, 1], 1),
        np.log(np.maximum(gt[:, 2], 1) / np.maximum(visual_pred[:, 2], 1)),
        np.log(np.maximum(gt[:, 3], 1) / np.maximum(visual_pred[:, 3], 1)),
    ], axis=1)
    return gate, np.clip(correction / 0.25, -1, 1).astype(np.float32)


def _apply(
    base: np.ndarray, visual: np.ndarray, gate_prob: np.ndarray,
    correction: np.ndarray, threshold: float, alpha: float,
) -> np.ndarray:
    use_visual = (gate_prob >= threshold).astype(np.float32)[:, None]
    chosen = base + use_visual * (visual - base)
    out = chosen.copy()
    out[:, 0] += alpha * 0.25 * correction[:, 0] * base[:, 2]
    out[:, 1] += alpha * 0.25 * correction[:, 1] * base[:, 3]
    out[:, 2] *= np.exp(alpha * 0.25 * correction[:, 2])
    out[:, 3] *= np.exp(alpha * 0.25 * correction[:, 3])
    return out


@torch.no_grad()
def _infer(model, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(torch.from_numpy(x).float(), batch_size=1024)
    gates, corrections = [], []
    for batch in loader:
        gate, correction = model(batch.to(device))
        gates.append(torch.sigmoid(gate).cpu().numpy())
        corrections.append(correction.cpu().numpy())
    return np.concatenate(gates), np.concatenate(corrections)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="artifacts/candidate_dataset.npz")
    ap.add_argument("--visual", default="artifacts/visual_oof_predictions_dynamic.csv")
    ap.add_argument("--out", default="artifacts/temporal_safety_oof_predictions.csv")
    args = ap.parse_args()
    arrays = load_arrays(Path(args.dataset))
    visual = pd.read_csv(args.visual)
    expected = list(zip(map(str, arrays["domain"]), map(str, arrays["sequence"]), arrays["frame_idx"]))
    observed = list(zip(visual.domain.astype(str), visual.sequence.astype(str), visual.frame_idx))
    if expected != observed:
        raise RuntimeError("Visual OOF rows are not aligned with candidate dataset")
    current, visual_pred, _ = _current_vectors(arrays, visual)
    history = _history_windows(current, arrays)
    base, gt = arrays["boxes"][:, 0], arrays["gt"]
    gate_target, correction_target = _targets(visual_pred, base, gt)
    folds = build_dual_oof_folds(arrays)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, manifest = [], []
    for fold_number, fold in enumerate(folds):
        seed = 20260717 + 50000 + fold_number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_idx = fold.train_idx
        mean = history[train_idx].reshape(-1, history.shape[-1]).mean(0)
        std = history[train_idx].reshape(-1, history.shape[-1]).std(0)
        std[std < 1e-6] = 1
        scaled = ((history - mean[None, None]) / std[None, None]).astype(np.float32)
        model = TemporalSafetySizeNet(history.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        dataset = TensorDataset(
            torch.from_numpy(scaled[train_idx]),
            torch.from_numpy(gate_target[train_idx]),
            torch.from_numpy(correction_target[train_idx]),
        )
        sample_weights = sequence_balanced_weights(arrays, train_idx)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights), len(train_idx), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        loader = DataLoader(dataset, batch_size=512, sampler=sampler)
        positive = max(gate_target[train_idx].mean(), 1e-3)
        pos_weight = torch.tensor((1 - positive) / positive, device=device)
        for _ in range(25):
            model.train()
            for x, gate_y, correction_y in loader:
                optimizer.zero_grad(set_to_none=True)
                gate, correction = model(x.to(device))
                gate_loss = nn.functional.binary_cross_entropy_with_logits(
                    gate, gate_y.to(device), pos_weight=pos_weight
                )
                size_weight = 1.5 if fold.target_domain == "trackves" else 1.0
                correction_loss = nn.functional.smooth_l1_loss(
                    correction, correction_y.to(device)
                )
                loss = gate_loss + 0.35 * size_weight * correction_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        val_idx = fold.val_idx[arrays["domain"][fold.val_idx] == fold.target_domain]
        val_gate, val_correction = _infer(model, scaled[val_idx], device)
        best = (-1.0, 0.5, 0.0)
        for threshold in np.linspace(0.30, 0.80, 11):
            for alpha in np.linspace(0.0, 1.0, 11):
                pred = _apply(
                    base[val_idx], visual_pred[val_idx], val_gate, val_correction,
                    float(threshold), float(alpha),
                )
                score = float(_np_iou(pred, gt[val_idx]).mean())
                if score > best[0]:
                    best = (score, float(threshold), float(alpha))
        test_gate, test_correction = _infer(model, scaled[fold.test_idx], device)
        pred = _apply(
            base[fold.test_idx], visual_pred[fold.test_idx],
            test_gate, test_correction, best[1], best[2],
        )
        weights = np.zeros((len(fold.test_idx), 7), dtype=np.float32)
        weights[:, 0] = 1 - (test_gate >= best[1]).astype(np.float32)
        visual_weights = np.stack(
            visual.iloc[fold.test_idx].weights.map(json.loads)
        ).astype(np.float32)
        weights += (test_gate >= best[1])[:, None] * visual_weights
        rows.extend(_prediction_rows(
            arrays, fold.test_idx, "temporal_safety_size", pred, weights, fold.fold_id
        ))
        manifest.append({
            "fold_id": fold.fold_id, "threshold": best[1],
            "correction_alpha": best[2], "val_iou": best[0],
            "train_sequences": sorted(set(map(str, arrays["sequence"][train_idx]))),
        })
        print(f"[{fold_number + 1}/{len(folds)}] {fold.fold_id}", flush=True)
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".manifest.json").write_text(
        json.dumps({"history": HISTORY, "folds": manifest}, indent=2), encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()

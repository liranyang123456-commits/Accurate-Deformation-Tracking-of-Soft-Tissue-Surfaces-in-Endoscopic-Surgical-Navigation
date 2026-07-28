"""Leakage-safe second-stage temporal gate over first-stage OOF qualities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_exporter import box_iou
from global_redetection_gate_oof import _features, _iou_vector


def _temporal_features(
    sequence: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(sequence), 12), dtype=np.float32)
    for name in np.unique(sequence):
        indices = np.flatnonzero(sequence == name)
        nano, template = boxes[indices, 0], boxes[indices, 1]
        nc = nano[:, :2] + nano[:, 2:] / 2
        tc = template[:, :2] + template[:, 2:] / 2
        disagreement = np.linalg.norm(nc - tc, axis=1) / np.maximum(
            np.linalg.norm(nano[:, 2:], axis=1), 1.0
        )
        template_motion = np.r_[0.0, np.linalg.norm(np.diff(tc, axis=0), axis=1)] / np.maximum(
            np.linalg.norm(template[:, 2:], axis=1), 1.0
        )
        nano_motion = np.r_[0.0, np.linalg.norm(np.diff(nc, axis=0), axis=1)] / np.maximum(
            np.linalg.norm(nano[:, 2:], axis=1), 1.0
        )
        base = np.stack([
            disagreement, template_motion, nano_motion, scores[indices, 1],
        ], axis=1)
        columns = [base]
        for window in (3, 8):
            rolling = np.zeros_like(base)
            for position in range(len(indices)):
                rolling[position] = base[max(0, position - window + 1):position + 1].mean(axis=0)
            columns.append(rolling)
        output[indices] = np.concatenate(columns, axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/candidate_dataset.npz"))
    parser.add_argument("--redetection", type=Path, default=Path("artifacts/global_redetection_all.npz"))
    parser.add_argument("--visual-oof", type=Path, default=Path("artifacts/global_visual_gate_oof.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_temporal_gate_oof.json"))
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    redetection = np.load(args.redetection, allow_pickle=False)
    first_stage = np.load(args.visual_oof, allow_pickle=False)
    indices = redetection["indices"].astype(int)
    boxes = redetection["boxes"].astype(np.float32)
    scores = redetection["scores"].astype(np.float32)
    sequence = data["sequence"][indices].astype(str)
    gt = data["gt"][indices]
    frame_idx = data["frame_idx"][indices].astype(np.float32)
    nano_score = data["features"][indices, 1, 9].astype(np.float32)
    quality = first_stage["predicted_quality"].astype(np.float32)
    scalar = _features(boxes, scores, nano_score, frame_idx)
    x = np.concatenate([
        quality, quality - quality[:, :1], scalar,
        _temporal_features(sequence, boxes, scores),
    ], axis=1)
    targets = np.stack([
        _iou_vector(boxes[:, candidate], gt) for candidate in range(3)
    ], axis=1)
    names = sorted(np.unique(sequence))
    rows = []
    for fold, test_name in enumerate(names):
        val_name = names[(fold + 1) % len(names)]
        test = sequence == test_name
        validation = sequence == val_name
        train = ~(test | validation)
        mean, std = x[train].mean(0), x[train].std(0)
        std[std < 1e-6] = 1.0
        normalized = (x - mean) / std
        weights = np.zeros(int(train.sum()), dtype=np.float32)
        train_sequence = sequence[train]
        for name in np.unique(train_sequence):
            mask = train_sequence == name
            weights[mask] = 1.0 / max(int(mask.sum()), 1)
        val_quality = np.zeros((int(validation.sum()), 3), dtype=np.float32)
        test_quality = np.zeros((int(test.sum()), 3), dtype=np.float32)
        for candidate in range(3):
            model = HistGradientBoostingRegressor(
                learning_rate=0.04, max_iter=220, max_leaf_nodes=15,
                min_samples_leaf=35, l2_regularization=0.4,
                random_state=20260718,
            )
            model.fit(normalized[train], targets[train, candidate], sample_weight=weights)
            val_quality[:, candidate] = model.predict(normalized[validation])
            test_quality[:, candidate] = model.predict(normalized[test])
        val_winner = val_quality.argmax(1)
        val_gain = val_quality[np.arange(len(val_winner)), val_winner] - val_quality[:, 0]
        best_threshold, best_score = 0.0, -1.0
        for threshold in np.linspace(0.0, 0.35, 36):
            choice = np.where(val_gain > threshold, val_winner, 0)
            score = targets[validation][np.arange(len(choice)), choice].mean()
            if score > best_score:
                best_score, best_threshold = float(score), float(threshold)
        winner = test_quality.argmax(1)
        gain = test_quality[np.arange(len(winner)), winner] - test_quality[:, 0]
        choice = np.where(gain > best_threshold, winner, 0)
        chosen = targets[test][np.arange(len(choice)), choice]
        rows.append({
            "sequence": test_name, "threshold": best_threshold,
            "nano_iou": float(targets[test, 0].mean()),
            "gated_iou": float(chosen.mean()),
            "template_rate": float(np.mean(choice == 1)),
            "orb_rate": float(np.mean(choice == 2)),
        })
    report = {
        "protocol": "stacked OOF visual qualities + sequence LOSO temporal gate",
        "macro_nano_iou": float(np.mean([row["nano_iou"] for row in rows])),
        "macro_gated_iou": float(np.mean([row["gated_iou"] for row in rows])),
        "sequences": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

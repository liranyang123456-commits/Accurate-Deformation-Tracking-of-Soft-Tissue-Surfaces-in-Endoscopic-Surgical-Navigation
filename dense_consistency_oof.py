"""Train a dense-frame GRU motion-consistency model with sparse-GT OOF tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from candidate_exporter import box_iou
from mask_targets import parse_polygons


HISTORY = 5


class MotionGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(5, 24, batch_first=True)
        self.head = nn.Sequential(nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 4))

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1])


def _bbox(poly):
    return np.asarray(cv2.boundingRect(np.round(poly).astype(np.int32)), dtype=np.float32)


def _deltas(boxes, scores):
    previous, current = boxes[:-1], boxes[1:]
    pc = previous[:, :2] + previous[:, 2:] / 2
    cc = current[:, :2] + current[:, 2:] / 2
    delta = np.stack([
        (cc[:, 0] - pc[:, 0]) / np.maximum(previous[:, 2], 1),
        (cc[:, 1] - pc[:, 1]) / np.maximum(previous[:, 3], 1),
        np.log(np.maximum(current[:, 2], 1) / np.maximum(previous[:, 2], 1)),
        np.log(np.maximum(current[:, 3], 1) / np.maximum(previous[:, 3], 1)),
        scores[1:],
    ], axis=1).astype(np.float32)
    return delta


def _windows(delta):
    x, y = [], []
    for i in range(HISTORY, len(delta)):
        x.append(delta[i - HISTORY:i])
        y.append(delta[i, :4])
    return np.asarray(x, np.float32), np.asarray(y, np.float32)


def _predict_boxes(model, boxes, scores, device):
    delta = _deltas(boxes, scores)
    output = boxes.copy()
    if len(delta) <= HISTORY:
        return output
    x, _ = _windows(delta)
    loader = DataLoader(torch.from_numpy(x), batch_size=1024)
    model.eval()
    with torch.no_grad():
        pred_delta = np.concatenate([model(batch.to(device)).cpu().numpy() for batch in loader])
    for local, d in enumerate(pred_delta, start=HISTORY + 1):
        previous = boxes[local - 1]
        pc = previous[:2] + previous[2:] / 2
        w = previous[2] * np.exp(np.clip(d[2], -0.25, 0.25))
        h = previous[3] * np.exp(np.clip(d[3], -0.25, 0.25))
        center = pc + np.asarray([d[0] * previous[2], d[1] * previous[3]])
        output[local] = [center[0] - w / 2, center[1] - h / 2, w, h]
    return output


def main():
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    dense_root = Path("artifacts/dense_trackves")
    sequences = ["EV1", "EV2", "EV3", "IV1", "IV2", "IV3", "IV4", "IV5", "IV6"]
    data = {}
    for sequence in sequences:
        frame = pd.read_csv(dense_root / f"{sequence}_dense_nano.csv")
        data[sequence] = {
            "frame": frame,
            "boxes": frame[["pseudo_x", "pseudo_y", "pseudo_w", "pseudo_h"]].to_numpy(np.float32),
            "scores": frame.nano_score.to_numpy(np.float32),
        }
    gt_root = Path(cfg["paths"]["trackves_root"])
    gt = {
        sequence: {idx: _bbox(poly) for idx, poly in parse_polygons(gt_root / sequence / "GT.xml").items()}
        for sequence in sequences
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, manifest = [], []
    for fold, test_sequence in enumerate(sequences):
        val_sequence = sequences[(fold + 1) % len(sequences)]
        train_sequences = [x for x in sequences if x not in {test_sequence, val_sequence}]
        train_x, train_y = [], []
        for sequence in train_sequences:
            x, y = _windows(_deltas(data[sequence]["boxes"], data[sequence]["scores"]))
            train_x.append(x)
            train_y.append(y)
        dataset = TensorDataset(
            torch.from_numpy(np.concatenate(train_x)),
            torch.from_numpy(np.concatenate(train_y)),
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        torch.manual_seed(20260717 + fold)
        model = MotionGRU().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(10):
            model.train()
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(x.to(device))
                loss = nn.functional.smooth_l1_loss(pred, y.to(device))
                loss.backward()
                optimizer.step()
        predicted = {
            sequence: _predict_boxes(
                model, data[sequence]["boxes"], data[sequence]["scores"], device
            )
            for sequence in (val_sequence, test_sequence)
        }
        val_frame = data[val_sequence]["frame"]
        best = (-1.0, 0.0)
        for alpha in np.linspace(0, 1, 11):
            scores = []
            for idx, target in gt[val_sequence].items():
                position = np.flatnonzero(val_frame.frame_idx.to_numpy() == idx)
                if len(position):
                    j = int(position[0])
                    blend = data[val_sequence]["boxes"][j] + alpha * (
                        predicted[val_sequence][j] - data[val_sequence]["boxes"][j]
                    )
                    scores.append(box_iou(blend, target))
            candidate = float(np.mean(scores))
            if candidate > best[0]:
                best = (candidate, float(alpha))
        test_frame = data[test_sequence]["frame"]
        for idx, target in gt[test_sequence].items():
            position = np.flatnonzero(test_frame.frame_idx.to_numpy() == idx)
            if not len(position):
                continue
            j = int(position[0])
            nano = data[test_sequence]["boxes"][j]
            pred = nano + best[1] * (predicted[test_sequence][j] - nano)
            rows.append({
                "fold_id": f"trackves_{test_sequence}",
                "sequence": test_sequence, "frame_idx": idx,
                "bbox_iou": box_iou(pred, target),
                "nano_iou": box_iou(nano, target),
            })
        manifest.append({
            "test": test_sequence, "validation": val_sequence,
            "alpha": best[1], "val_iou": best[0],
        })
        print(f"[{fold + 1}/9] {test_sequence}", flush=True)
    output = Path("artifacts/dense_consistency_oof.csv")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "folds": manifest,
        "motion_gru_iou": float(np.mean([x["bbox_iou"] for x in rows])),
        "nano_iou": float(np.mean([x["nano_iou"] for x in rows])),
        "uses_intermediate_gt": False,
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

"""Train/evaluate a lightweight real-polygon ROI mask head on TrackVes OOF."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset_builder import build_dual_oof_folds, load_arrays, sequence_balanced_weights


class LightMaskHead(nn.Module):
    def __init__(self, channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 1, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _iou(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    inter = np.logical_and(pred, target).sum((1, 2))
    union = np.logical_or(pred, target).sum((1, 2))
    return inter / np.maximum(union, 1)


@torch.no_grad()
def _predict(model, maps, indices, device):
    loader = DataLoader(torch.from_numpy(maps[indices]).float().div(255), batch_size=512)
    model.eval()
    return np.concatenate([
        torch.sigmoid(model(batch.to(device))).cpu().numpy() for batch in loader
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="artifacts/candidate_dataset.npz")
    ap.add_argument("--responses", default="artifacts/visual_responses_dynamic.npz")
    ap.add_argument("--masks", default="artifacts/trackves_roi_masks.npz")
    args = ap.parse_args()
    arrays = load_arrays(Path(args.dataset))
    maps = np.load(args.responses, allow_pickle=False)["maps"][:, 0]
    target_data = np.load(args.masks, allow_pickle=False)
    targets, valid = target_data["masks"], target_data["valid"].astype(bool)
    folds = build_dual_oof_folds(arrays)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, manifest = [], []
    for fold_number, fold in enumerate(folds):
        train_idx = fold.train_idx[
            (arrays["domain"][fold.train_idx] == fold.target_domain) & valid[fold.train_idx]
        ]
        val_idx = fold.val_idx[
            (arrays["domain"][fold.val_idx] == fold.target_domain) & valid[fold.val_idx]
        ]
        test_idx = fold.test_idx[valid[fold.test_idx]]
        model = LightMaskHead(maps.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        dataset = TensorDataset(
            torch.from_numpy(maps[train_idx]).float().div(255),
            torch.from_numpy(targets[train_idx]).float(),
        )
        weights = sequence_balanced_weights(arrays, train_idx)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), len(train_idx), replacement=True,
            generator=torch.Generator().manual_seed(20260717 + fold_number),
        )
        loader = DataLoader(dataset, batch_size=256, sampler=sampler)
        for _ in range(18):
            model.train()
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x.to(device))
                y = y.to(device)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
                prob = torch.sigmoid(logits)
                dice = 1 - (
                    (2 * (prob * y).sum((1, 2)) + 1)
                    / (prob.sum((1, 2)) + y.sum((1, 2)) + 1)
                ).mean()
                loss = bce + dice
                loss.backward()
                optimizer.step()
        val_prob = _predict(model, maps, val_idx, device)
        threshold_scores = []
        for threshold in np.linspace(0.25, 0.75, 11):
            threshold_scores.append((
                float(_iou(val_prob >= threshold, targets[val_idx]).mean()),
                float(threshold),
            ))
        _, threshold = max(threshold_scores)
        test_prob = _predict(model, maps, test_idx, device)
        learned_iou = _iou(test_prob >= threshold, targets[test_idx])
        rectangle = np.zeros((24, 24), dtype=bool)
        rectangle[4:20, 4:20] = True
        baseline_iou = _iou(
            np.repeat(rectangle[None], len(test_idx), axis=0), targets[test_idx]
        )
        for j, liou, biou in zip(test_idx, learned_iou, baseline_iou):
            rows.append({
                "fold_id": fold.fold_id, "sequence": str(arrays["sequence"][j]),
                "domain": str(arrays["domain"][j]),
                "frame_idx": int(arrays["frame_idx"][j]),
                "roi_mask_iou": float(liou), "bbox_mask_iou": float(biou),
            })
        checkpoint = Path("artifacts/mask_checkpoints") / fold.fold_id / "mask_head.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "threshold": threshold}, checkpoint)
        manifest.append({
            "fold_id": fold.fold_id, "threshold": threshold,
            "test_samples": len(test_idx),
        })
        print(f"[{fold_number + 1}/{len(folds)}] {fold.fold_id}", flush=True)
    output = Path("artifacts/mask_oof_predictions.csv")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = np.asarray([r["roi_mask_iou"] for r in rows])
    baseline = np.asarray([r["bbox_mask_iou"] for r in rows])
    by_domain = {}
    for domain in ("trackves", "chess"):
        selected = [r for r in rows if r["domain"] == domain]
        by_domain[domain] = {
            "roi_mask_iou": float(np.mean([r["roi_mask_iou"] for r in selected])),
            "bbox_mask_iou": float(np.mean([r["bbox_mask_iou"] for r in selected])),
            "samples": len(selected),
        }
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "folds": manifest, "roi_mask_iou": float(frame.mean()),
        "bbox_mask_iou": float(baseline.mean()),
        "by_domain": by_domain,
        "real_polygon_gt": True,
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

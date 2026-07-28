"""Run 9 TrackVes + 11 chess grouped OOF folds for all learned fusers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from candidate_exporter import box_iou, build_from_config
from dataset_builder import build_dual_oof_folds, load_arrays
from train_fold import train_one


def _center_error(a: np.ndarray, b: np.ndarray) -> float:
    ac = a[:2] + a[2:] / 2
    bc = b[:2] + b[2:] / 2
    return float(np.linalg.norm(ac - bc))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_rows(
    arrays: dict[str, np.ndarray], indices: np.ndarray, model_name: str,
    pred: np.ndarray, weights: np.ndarray, fold_id: str,
) -> list[dict]:
    rows = []
    for local_i, data_i in enumerate(indices):
        p, gt = pred[local_i], arrays["gt"][data_i]
        iou = box_iou(p, gt)
        rows.append({
            "fold_id": fold_id, "model": model_name,
            "domain": str(arrays["domain"][data_i]),
            "sequence": str(arrays["sequence"][data_i]),
            "unit": str(arrays["unit"][data_i]),
            "frame_idx": int(arrays["frame_idx"][data_i]),
            "pred_x": float(p[0]), "pred_y": float(p[1]),
            "pred_w": float(p[2]), "pred_h": float(p[3]),
            "gt_x": float(gt[0]), "gt_y": float(gt[1]),
            "gt_w": float(gt[2]), "gt_h": float(gt[3]),
            "bbox_iou": iou, "center_error_px": _center_error(p, gt),
            "success_05": int(iou >= 0.5),
            "precision_20": int(_center_error(p, gt) <= 20.0),
            "weights": json.dumps(weights[local_i].tolist(), separators=(",", ":")),
        })
    return rows


def _baseline_predictions(
    arrays: dict[str, np.ndarray], indices: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    boxes = arrays["boxes"][indices]
    k = boxes.shape[1]
    weights = np.zeros((len(indices), k), dtype=np.float32)
    if name == "full_fusion":
        choice = np.zeros(len(indices), dtype=int)
    elif name == "nano":
        choice = np.ones(len(indices), dtype=int)
    elif name == "hist":
        choice = np.full(len(indices), 2, dtype=int)
    elif name == "heuristic_score":
        # Score feature is index 8; disallow previous-frame candidate.
        score = arrays["features"][indices, :, 8].copy()
        score[:, 3] = -np.inf
        choice = score.argmax(axis=1)
    elif name == "domain_router":
        # Validation-free conservative rule motivated by source specialization:
        # Nano on TrackVes, frozen full fusion on chess.
        choice = np.where(arrays["domain"][indices] == "trackves", 1, 0)
    elif name == "oracle":
        gt = arrays["gt"][indices]
        all_ious = np.asarray([
            [box_iou(candidate, target) for candidate in sample_boxes]
            for sample_boxes, target in zip(boxes, gt)
        ])
        choice = all_ious.argmax(axis=1)
    else:
        raise ValueError(name)
    weights[np.arange(len(indices)), choice] = 1.0
    return boxes[np.arange(len(indices)), choice], weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--rebuild-dataset", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Run two outer folds")
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = root / cfg["paths"]["artifacts"]
    dataset_path = artifacts / "candidate_dataset.npz"
    if args.rebuild_dataset or not dataset_path.is_file():
        build_from_config(config_path)
    arrays = load_arrays(dataset_path)
    folds = build_dual_oof_folds(arrays)
    if args.smoke:
        folds = [next(f for f in folds if f.fold_id == "trackves_EV1"),
                 next(f for f in folds if f.target_domain == "chess")]
    checkpoints = artifacts / "checkpoints"
    rows: list[dict] = []
    training_meta = []
    for fold_number, fold in enumerate(folds):
        split_record = {
            "fold_id": fold.fold_id, "target_domain": fold.target_domain,
            "test": [[fold.target_domain, fold.test_sequence]],
            "validation": [list(x) for x in fold.val_sequences],
            "train_count": len(fold.train_idx), "val_count": len(fold.val_idx),
            "test_count": len(fold.test_idx),
        }
        # Explicit leakage assertion at sequence/domain level.
        train_pairs = set(zip(
            map(str, arrays["domain"][fold.train_idx]),
            map(str, arrays["sequence"][fold.train_idx]),
        ))
        assert (fold.target_domain, fold.test_sequence) not in train_pairs
        assert not train_pairs.intersection(set(fold.val_sequences))
        training_meta.append({"split": split_record, "models": []})
        for model_name in ("linear", "tiny_mlp", "micro_transformer"):
            ckpt = checkpoints / fold.fold_id / f"{model_name}.pt"
            meta, pred, weights = train_one(
                model_name=model_name, fold=fold, raw_arrays=arrays, cfg=cfg,
                checkpoint_path=ckpt, seed_offset=fold_number * 10,
            )
            meta["checkpoint"] = str(ckpt.relative_to(root))
            meta["checkpoint_sha256"] = _sha256(ckpt)
            training_meta[-1]["models"].append(meta)
            rows.extend(_prediction_rows(
                arrays, fold.test_idx, model_name, pred, weights, fold.fold_id
            ))
        for baseline in (
            "full_fusion", "nano", "hist", "heuristic_score", "domain_router", "oracle"
        ):
            pred, weights = _baseline_predictions(arrays, fold.test_idx, baseline)
            rows.extend(_prediction_rows(
                arrays, fold.test_idx, baseline, pred, weights, fold.fold_id
            ))
        print(f"[{fold_number + 1}/{len(folds)}] {fold.fold_id} complete", flush=True)
    predictions_path = artifacts / ("oof_predictions_smoke.csv" if args.smoke else "oof_predictions.csv")
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.executable + " " + " ".join(sys.argv),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "seed": cfg["seed"],
        "config": cfg, "config_sha256": _sha256(config_path),
        "dataset_sha256": _sha256(dataset_path), "fold_count": len(folds),
        "gt_in_inference_features": False,
        "protocol": "sequence-level dual-domain OOF",
        "training": training_meta,
    }
    manifest_path = artifacts / ("run_manifest_smoke.json" if args.smoke else "run_manifest.json")
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(predictions_path)


if __name__ == "__main__":
    main()

"""Run the visual quality + continuous residual model on all 20 OOF folds."""
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

from dataset_builder import build_dual_oof_folds, load_arrays
from run_nested_oof import _prediction_rows
from train_visual_fold import train_visual_one
from visual_feature_exporter import export
from visual_feature_exporter_dynamic import export_dynamic


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target-only", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--experiment-tag", default="")
    ap.add_argument("--size-weight", type=float, default=None)
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.size_weight is not None:
        cfg["visual_training"]["trackves_size_weight"] = float(args.size_weight)
    artifacts = root / cfg["paths"]["artifacts"]
    candidate_path = artifacts / "candidate_dataset.npz"
    visual_path = artifacts / (
        "visual_responses_dynamic.npz" if args.dynamic else "visual_responses.npz"
    )
    if not visual_path.is_file():
        export_dynamic(config_path) if args.dynamic else export(config_path)
    arrays = load_arrays(candidate_path)
    visual = np.load(visual_path, allow_pickle=False)
    maps, valid = visual["maps"], visual["valid"]
    if len(maps) != len(arrays["gt"]):
        raise RuntimeError("Visual response count does not match candidate dataset")
    folds = build_dual_oof_folds(arrays)
    if args.smoke:
        folds = [folds[0], next(f for f in folds if f.target_domain == "chess")]
    rows, training = [], []
    suffix = ("_dynamic" if args.dynamic else "") + (
        "_target_only" if args.target_only else ""
    )
    if args.experiment_tag:
        suffix += "_" + args.experiment_tag.strip().replace("-", "_")
    model_label = (
        "visual_dynamic_target_only" if args.dynamic and args.target_only
        else "visual_dynamic" if args.dynamic
        else "visual_target_only" if args.target_only
        else "visual_quality_residual"
    )
    if args.experiment_tag:
        model_label += "_" + args.experiment_tag.strip().replace("-", "_")
    checkpoints = artifacts / f"visual_checkpoints{suffix}"
    for fold_number, fold in enumerate(folds):
        ckpt = checkpoints / fold.fold_id / f"{model_label}.pt"
        meta, pred, weights, quality = train_visual_one(
            fold=fold, raw_arrays=arrays, visual_maps=maps, visual_valid=valid,
            cfg=cfg, checkpoint_path=ckpt, seed_offset=fold_number * 10,
            target_only=args.target_only, model_label=model_label,
        )
        meta["checkpoint"] = str(ckpt.relative_to(root))
        meta["checkpoint_sha256"] = _sha256(ckpt)
        split = {
            "fold_id": fold.fold_id, "target_domain": fold.target_domain,
            "test": [[fold.target_domain, fold.test_sequence]],
            "validation": [list(x) for x in fold.val_sequences],
            "train_count": len(fold.train_idx), "val_count": len(fold.val_idx),
            "test_count": len(fold.test_idx),
        }
        training.append({"split": split, "model": meta})
        fold_rows = _prediction_rows(
            arrays, fold.test_idx, model_label,
            pred, weights, fold.fold_id,
        )
        for row, q in zip(fold_rows, quality):
            row["predicted_candidate_quality"] = json.dumps(
                q.tolist(), separators=(",", ":")
            )
        rows.extend(fold_rows)
        print(f"[{fold_number + 1}/{len(folds)}] {fold.fold_id} complete", flush=True)
    stem = f"visual_oof_predictions{suffix}"
    output = artifacts / (f"{stem}_smoke.csv" if args.smoke else f"{stem}.csv")
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.executable + " " + " ".join(sys.argv),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": cfg["seed"], "fold_count": len(folds),
        "candidate_dataset_sha256": _sha256(candidate_path),
        "visual_responses_sha256": _sha256(visual_path),
        "valid_visual_samples": int(valid.sum()),
        "gt_in_inference_features": False,
        "first_frame_gt_template_only": True,
        "training": training,
    }
    manifest_stem = f"visual_run_manifest{suffix}"
    manifest_path = artifacts / (
        f"{manifest_stem}_smoke.json" if args.smoke else f"{manifest_stem}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

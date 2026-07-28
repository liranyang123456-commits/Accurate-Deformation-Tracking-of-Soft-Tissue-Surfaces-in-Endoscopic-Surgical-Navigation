"""Export sequential every-frame Nano pseudo labels for consistency training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from candidate_exporter import box_iou
from mask_targets import parse_polygons
from visual_feature_exporter import _find_video


def _polygon_bbox(polygon: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(np.round(polygon).astype(np.int32))
    return np.asarray([x, y, w, h], dtype=np.float32)


def run_sequence(root: Path, sequence: str, output: Path) -> dict:
    polygons = parse_polygons(root / sequence / "GT.xml")
    gt_boxes = {frame: _polygon_bbox(poly) for frame, poly in polygons.items()}
    first_frame = min(gt_boxes)
    last_frame = max(gt_boxes)
    cap = cv2.VideoCapture(str(_find_video(root, sequence)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot decode {sequence}:{first_frame}")
    params = cv2.TrackerNano_Params()
    params.backbone = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
    params.neckhead = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"
    tracker = cv2.TrackerNano_create(params)
    tracker.init(image, tuple(map(int, gt_boxes[first_frame])))
    previous = gt_boxes[first_frame].copy()
    rows = []
    for frame_idx in range(first_frame, last_frame + 1):
        if frame_idx != first_frame:
            ok, image = cap.read()
            if not ok:
                break
            success, raw = tracker.update(image)
            if success:
                previous = np.asarray(raw, dtype=np.float32)
        score = float(getattr(tracker, "getTrackingScore", lambda: 1.0)())
        gt = gt_boxes.get(frame_idx)
        rows.append({
            "sequence": sequence, "frame_idx": frame_idx,
            "pseudo_x": float(previous[0]), "pseudo_y": float(previous[1]),
            "pseudo_w": float(previous[2]), "pseudo_h": float(previous[3]),
            "nano_score": score, "is_gt_frame": int(gt is not None),
            "gt_iou": "" if gt is None else box_iou(previous, gt),
        })
    cap.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "sequence": sequence, "frames": len(rows),
        "gt_frames": sum(r["is_gt_frame"] for r in rows),
        "mean_gt_iou": float(np.mean([
            r["gt_iou"] for r in rows if r["gt_iou"] != ""
        ])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--sequences", default="EV1")
    args = ap.parse_args()
    config = Path(args.config).resolve()
    project = config.parent.parent
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["trackves_root"])
    output_root = project / cfg["paths"]["artifacts"] / "dense_trackves"
    summaries = []
    for sequence in args.sequences.split(","):
        sequence = sequence.strip()
        if sequence:
            summaries.append(run_sequence(
                root, sequence, output_root / f"{sequence}_dense_nano.csv"
            ))
    (output_root / "manifest.json").write_text(
        json.dumps({"summaries": summaries, "gt_used_for_initialization_only": True,
                    "intermediate_frames_are_pseudo_labels": True}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

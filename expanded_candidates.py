"""Generate multi-direction/multi-scale local candidates and measure headroom."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from candidate_exporter import box_iou
from dataset_builder import load_arrays


def expand(boxes: np.ndarray) -> np.ndarray:
    output = [box.copy() for box in boxes]
    for source in boxes[:3]:
        x, y, w, h = source
        for dx, dy in ((-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)):
            candidate = source.copy()
            candidate[0] += dx * w
            candidate[1] += dy * h
            output.append(candidate)
        for scale in (0.85, 1.15):
            candidate = source.copy()
            cx, cy = x + w / 2, y + h / 2
            candidate[2], candidate[3] = w * scale, h * scale
            candidate[0], candidate[1] = cx - candidate[2] / 2, cy - candidate[3] / 2
            output.append(candidate)
    return np.stack(output).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="artifacts/candidate_dataset.npz")
    args = ap.parse_args()
    arrays = load_arrays(Path(args.dataset))
    rows = []
    for domain in ("trackves", "chess"):
        mask = arrays["domain"] == domain
        base, original_oracle, expanded_oracle = [], [], []
        for boxes, gt in zip(arrays["boxes"][mask], arrays["gt"][mask]):
            base.append(box_iou(boxes[0], gt))
            original_oracle.append(max(box_iou(x, gt) for x in boxes))
            expanded_oracle.append(max(box_iou(x, gt) for x in expand(boxes)))
        rows.append({
            "domain": domain, "candidates": int(len(expand(arrays["boxes"][mask][0]))),
            "base_iou": float(np.mean(base)),
            "original_oracle_iou": float(np.mean(original_oracle)),
            "expanded_oracle_iou": float(np.mean(expanded_oracle)),
        })
    output = Path("artifacts/expanded_candidate_oracle.json")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

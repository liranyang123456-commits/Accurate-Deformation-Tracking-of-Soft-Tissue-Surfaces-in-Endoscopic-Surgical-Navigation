"""Export real TrackVes polygon masks into frozen-final ROI coordinates."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

from dataset_builder import load_arrays


SIZE = 24


def _read_matrix(node: ET.Element) -> np.ndarray:
    rows = int(node.findtext("rows", "0"))
    cols = int(node.findtext("cols", "0"))
    values = np.asarray(
        [float(x) for x in node.findtext("data", "").split()], dtype=np.float32
    )
    return values.reshape(rows, cols) if rows * cols == values.size else values


def parse_polygons(path: Path) -> dict[int, np.ndarray]:
    polygons = {}
    for node in ET.parse(path).getroot():
        if not str(node.tag).startswith("GT"):
            continue
        try:
            frame = int(str(node.tag)[2:])
        except ValueError:
            continue
        polygon = _read_matrix(node)
        if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3:
            polygons[frame] = polygon.astype(np.float32)
    return polygons


def polygon_roi_mask(polygon: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = map(float, box)
    crop_w, crop_h = max(4.0, 1.5 * w), max(4.0, 1.5 * h)
    x1 = x + w / 2 - crop_w / 2
    y1 = y + h / 2 - crop_h / 2
    transformed = polygon.copy()
    transformed[:, 0] = (transformed[:, 0] - x1) * SIZE / crop_w
    transformed[:, 1] = (transformed[:, 1] - y1) * SIZE / crop_h
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(transformed).astype(np.int32)], 1)
    return mask


def full_mask_roi(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = map(float, box)
    crop_w, crop_h = max(4.0, 1.5 * w), max(4.0, 1.5 * h)
    x1 = max(0, int(round(x + w / 2 - crop_w / 2)))
    y1 = max(0, int(round(y + h / 2 - crop_h / 2)))
    x2 = min(mask.shape[1], int(round(x + w / 2 + crop_w / 2)))
    y2 = min(mask.shape[0], int(round(y + h / 2 + crop_h / 2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    return (
        cv2.resize(mask[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        > 0
    ).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    config = Path(args.config).resolve()
    root = config.parent.parent
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    artifacts = root / cfg["paths"]["artifacts"]
    arrays = load_arrays(artifacts / "candidate_dataset.npz")
    masks = np.zeros((len(arrays["gt"]), SIZE, SIZE), dtype=np.uint8)
    valid = np.zeros(len(arrays["gt"]), dtype=np.uint8)
    polygon_cache = {}
    trackves_root = Path(cfg["paths"]["trackves_root"])
    for i, (domain, sequence, frame) in enumerate(zip(
        arrays["domain"], arrays["sequence"], arrays["frame_idx"]
    )):
        if str(domain) == "chess":
            image_path = Path(str(arrays["source_path"][i]))
            mask_path = (
                Path(r"E:\Surgical_Tracking_Datasets\Chess_Box_Mask")
                / str(sequence) / "mask" / f"{image_path.stem}.png"
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[i] = full_mask_roi(mask, arrays["boxes"][i, 0])
                valid[i] = 1
            continue
        if str(domain) != "trackves":
            continue
        sequence = str(sequence)
        if sequence not in polygon_cache:
            polygon_cache[sequence] = parse_polygons(trackves_root / sequence / "GT.xml")
        polygon = polygon_cache[sequence].get(int(frame))
        if polygon is None:
            continue
        masks[i] = polygon_roi_mask(polygon, arrays["boxes"][i, 0])
        valid[i] = 1
    output = artifacts / "trackves_roi_masks.npz"
    np.savez_compressed(output, masks=masks, valid=valid, size=np.asarray(SIZE))
    (artifacts / "mask_manifest.json").write_text(json.dumps({
        "target": "real TrackVes GT polygon rasterized in frozen-final 1.5x ROI",
        "valid_samples": int(valid.sum()), "size": SIZE,
        "trackves_polygon_samples": int(
            valid[arrays["domain"] == "trackves"].sum()
        ),
        "chess_true_mask_samples": int(valid[arrays["domain"] == "chess"].sum()),
        "chess_mask_available": True,
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

"""Export six-channel static+adaptive-template candidate response maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from visual_feature_exporter import (
    MAP_SIZE, _FrameReader, _crop, _responses,
)


CHANNEL_NAMES = (
    "gray", "edge", "static_template_absdiff", "static_hsv_backprojection",
    "dynamic_template_absdiff", "dynamic_hsv_backprojection",
)


def _template_pair(patch: np.ndarray, template: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray, template_gray)
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_template], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    backprojection = cv2.calcBackProject(
        [hsv_patch], [0, 1], hist, [0, 180, 0, 256], 1
    )
    return np.stack([difference, backprojection]).astype(np.uint8)


def responses_dynamic(
    patch: np.ndarray, static_template: np.ndarray, dynamic_template: np.ndarray
) -> np.ndarray:
    static = _responses(patch, static_template)
    dynamic = _template_pair(patch, dynamic_template)
    return np.concatenate([static, dynamic], axis=0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_dynamic(config_path: Path, force: bool = False) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    artifacts = root / cfg["paths"]["artifacts"]
    candidate_path = artifacts / "candidate_dataset.npz"
    output = artifacts / "visual_responses_dynamic.npz"
    if output.is_file() and not force:
        return output
    data = np.load(candidate_path, allow_pickle=False)
    n, k = data["boxes"].shape[:2]
    maps = np.zeros((n, k, len(CHANNEL_NAMES), MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    valid = np.zeros(n, dtype=np.uint8)
    static_templates: dict[str, np.ndarray] = {}
    dynamic_templates: dict[str, np.ndarray] = {}
    reader = _FrameReader(Path(cfg["paths"]["trackves_root"]))
    try:
        for i in range(n):
            domain, sequence, unit = map(str, (
                data["domain"][i], data["sequence"][i], data["unit"][i]
            ))
            image = reader.read(
                domain, sequence, int(data["frame_idx"][i]), str(data["source_path"][i])
            )
            if image is None:
                continue
            key = f"{domain}:{unit}"
            if key not in static_templates:
                initial = _crop(image, data["gt"][i], context=1.0)
                static_templates[key] = initial
                dynamic_templates[key] = initial.copy()
            for candidate_idx, box in enumerate(data["boxes"][i]):
                patch = _crop(image, box)
                maps[i, candidate_idx] = responses_dynamic(
                    patch, static_templates[key], dynamic_templates[key]
                )
            # Update from the frozen final prediction only; never from GT.
            reliability = float(data["features"][i, 0, 16])
            risk = float(data["features"][i, 0, 15])
            if reliability >= 0.55 and risk <= 0.50:
                current = _crop(image, data["boxes"][i, 0], context=1.0)
                dynamic_templates[key] = cv2.addWeighted(
                    dynamic_templates[key], 0.90, current, 0.10, 0.0
                )
            valid[i] = 1
            if (i + 1) % 250 == 0:
                print(f"[{i + 1}/{n}] dynamic visual responses", flush=True)
    finally:
        reader.close()
    np.savez_compressed(
        output, maps=maps, valid=valid,
        channel_names=np.asarray(CHANNEL_NAMES), map_size=np.asarray(MAP_SIZE),
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": n, "valid_samples": int(valid.sum()),
        "channels": CHANNEL_NAMES, "map_size": MAP_SIZE,
        "candidate_dataset_sha256": _sha256(candidate_path),
        "visual_responses_sha256": _sha256(output),
        "dynamic_update": (
            "EMA 0.10 from frozen final prediction when reliability>=0.55 "
            "and risk<=0.50; no GT updates."
        ),
    }
    (artifacts / "visual_dynamic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(export_dynamic(Path(args.config).resolve(), args.force))

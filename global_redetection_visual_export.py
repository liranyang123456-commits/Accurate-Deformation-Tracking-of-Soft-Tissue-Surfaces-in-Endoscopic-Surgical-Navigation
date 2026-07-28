"""Export visual response maps for Nano/template/ORB global candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from visual_feature_exporter import _FrameReader, _crop, _responses


def _raw_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = map(int, np.round(box))
    patch = image[max(0, y):min(image.shape[0], y + h),
                  max(0, x):min(image.shape[1], x + w)]
    return cv2.resize(patch, (24, 24), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/candidate_dataset.npz"))
    parser.add_argument("--redetection", type=Path, default=Path("artifacts/global_redetection_all.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_redetection_visual.npz"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = np.load(args.dataset, allow_pickle=False)
    redetection = np.load(args.redetection, allow_pickle=False)
    indices = redetection["indices"].astype(int)
    boxes = redetection["boxes"]
    maps = np.zeros((len(indices), 3, 4, 24, 24), dtype=np.uint8)
    valid = np.zeros(len(indices), dtype=np.uint8)
    reader = _FrameReader(Path(cfg["paths"]["trackves_root"]))
    templates: dict[str, np.ndarray] = {}
    try:
        for position, index in enumerate(indices):
            sequence = str(data["sequence"][index])
            image = reader.read(
                "trackves", sequence, int(data["frame_idx"][index]), ""
            )
            if image is None:
                continue
            if sequence not in templates:
                templates[sequence] = _raw_crop(image, data["gt"][index])
            for candidate in range(3):
                maps[position, candidate] = _responses(
                    _crop(image, boxes[position, candidate]),
                    templates[sequence],
                )
            valid[position] = 1
            if (position + 1) % 250 == 0:
                print(f"[{position + 1}/{len(indices)}] global visual maps", flush=True)
    finally:
        reader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, maps=maps, valid=valid, indices=indices)
    print(args.output)


if __name__ == "__main__":
    main()

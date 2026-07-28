"""Validate genuinely independent external tracking sequences before ingestion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_manifest(path: Path, known_sequences: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "domain", "frames"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    name = str(payload["name"])
    if name in known_sequences:
        raise ValueError(f"Sequence {name!r} is not independent; name already exists")
    frames = payload["frames"]
    if len(frames) < 2:
        raise ValueError("An independent sequence needs at least two annotated frames")
    seen = set()
    for item in frames:
        image = Path(item["image"])
        if not image.is_file():
            raise FileNotFoundError(image)
        frame_idx = int(item["frame_idx"])
        if frame_idx in seen:
            raise ValueError(f"Duplicate frame_idx {frame_idx}")
        seen.add(frame_idx)
        polygon = np.asarray(item["polygon"], dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(f"Invalid polygon at frame {frame_idx}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "name": name, "domain": str(payload["domain"]),
        "annotated_frames": len(frames), "manifest_sha256": digest,
        "independent_name_check": True, "ready_for_loso": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument("--known", default="")
    ap.add_argument("--out", default="artifacts/external_sequence_validation.json")
    args = ap.parse_args()
    result = validate_manifest(
        Path(args.manifest).resolve(),
        {x.strip() for x in args.known.split(",") if x.strip()},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

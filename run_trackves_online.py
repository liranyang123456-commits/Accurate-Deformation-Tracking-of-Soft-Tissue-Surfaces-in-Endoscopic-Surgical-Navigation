"""Run the final online router under the original TrackVes evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trackves_eval_all_methods as benchmark
from online_tracker import FinalHybridOnlineTracker

TRACKER_KEY = "ours_hybrid_dynamic_online"


def _paths() -> tuple[Path, Path, Path]:
    return (
        HERE / "artifacts" / "deployment" / "hybrid_dynamic_chess.pt",
        HERE / "artifacts" / "deployment" / "chess_mask_head.pt",
        HERE / "config" / "default.yaml",
    )


def _install_factory() -> None:
    checkpoint, mask_checkpoint, config = _paths()
    original = benchmark._make_tracker

    def factory(name, init_image, init_bbox, init_mask, args):
        if name == TRACKER_KEY:
            return FinalHybridOnlineTracker(
                init_image, init_bbox, init_mask, "trackves",
                checkpoint, mask_checkpoint, config,
            )
        return original(name, init_image, init_bbox, init_mask, args)

    benchmark._make_tracker = factory
    benchmark.TRACKER_LABELS[TRACKER_KEY] = "Ours Hybrid Dynamic Online"


def _aggregate(summaries: list[dict], out_dir: Path) -> None:
    valid = [s for s in summaries if not s.get("error")]
    fields = [
        "sequence", "tracker", "tracker_label", "n_gt_frames",
        "bbox_iou_mean", "mask_iou_mean", "center_err_mean",
        "precision_rate", "success_rate", "elapsed_s", "sequential", "error",
    ]
    with (out_dir / "trackves_per_seq_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in summaries])
    metrics = [
        "bbox_iou_mean", "mask_iou_mean", "center_err_mean",
        "precision_rate", "success_rate",
    ]
    overall = {
        "tracker": TRACKER_KEY,
        "tracker_label": benchmark.TRACKER_LABELS[TRACKER_KEY],
        "n_sequences": len(valid),
        "n_gt_frames_total": sum(int(row.get("n_gt_frames", 0)) for row in valid),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in valid if np.isfinite(float(row.get(metric, np.nan)))]
        overall[metric] = float(np.mean(values)) if values else None
    with (out_dir / "trackves_online_results.json").open("w", encoding="utf-8") as stream:
        json.dump({"overall": overall, "sequences": summaries}, stream, ensure_ascii=False, indent=2)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(benchmark.TRACKVES_ROOT))
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "online_trackves")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    options = parser.parse_args()
    _install_factory()
    options.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip() for item in options.sequences.split(",") if item.strip()}
    sequences = [
        path for path in sorted(options.data_root.iterdir())
        if path.is_dir() and (not requested or path.name in requested)
        and (path / "GT.xml").exists()
    ]
    if options.max_sequences > 0:
        sequences = sequences[:options.max_sequences]
    eval_args = SimpleNamespace(
        proc_scale=1.0, save_vis=options.save_vis, sequential=False,
        print_every=20, save_intermediate=False,
        save_intermediate_overlay=False, force=options.force,
    )
    summaries = []
    for sequence in sequences:
        summaries.extend(
            benchmark.evaluate_sequence(
                sequence, [TRACKER_KEY], options.out_dir, eval_args
            )
        )
    _aggregate(summaries, options.out_dir)


if __name__ == "__main__":
    main()

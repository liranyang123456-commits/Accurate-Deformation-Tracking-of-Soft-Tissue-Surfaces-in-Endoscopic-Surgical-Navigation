"""Run the final online router on all chessboard sequences."""
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

import compare_chessboard_trackers as benchmark
from online_tracker import FinalHybridOnlineTracker

TRACKER_KEY = "ours_hybrid_dynamic_online"


def _install_factory() -> None:
    original = benchmark._make_tracker
    checkpoint = HERE / "artifacts" / "deployment" / "hybrid_dynamic_chess.pt"
    mask_checkpoint = HERE / "artifacts" / "deployment" / "chess_mask_head.pt"
    config = HERE / "config" / "default.yaml"

    def factory(name, init_image, init_bbox, init_mask, args):
        if name == TRACKER_KEY:
            return FinalHybridOnlineTracker(
                init_image, init_bbox, init_mask, "chess",
                checkpoint, mask_checkpoint, config,
            )
        return original(name, init_image, init_bbox, init_mask, args)

    benchmark._make_tracker = factory
    benchmark.TRACKER_LABELS[TRACKER_KEY] = "Ours Hybrid Dynamic Online"


def _collect(outputs: list[dict], out_dir: Path) -> None:
    rows = []
    for output in outputs:
        sequence = str(output.get("online_unit", Path(output["frames_dir"]).name))
        for group in output.get("boards", [output]):
            for result in group.get("trackers", []):
                rows.append({"sequence": sequence, **result})
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0])
        with (out_dir / "chess_online_per_sequence.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    valid = [row for row in rows if not row.get("error")]
    overall = {
        "tracker": TRACKER_KEY,
        "n_sequences": len({row["sequence"].split("/")[0] for row in valid}),
        "n_units": len({row["sequence"] for row in valid}),
        "n_groups": len(valid),
        "n_frames_total": sum(int(row["n_frames"]) for row in valid),
    }
    for metric in (
        "bbox_iou_mean", "mask_iou_mean", "center_error_mean_px",
        "precision_rate", "success_rate",
    ):
        values = [float(row[metric]) for row in valid if np.isfinite(float(row[metric]))]
        overall[metric] = float(np.mean(values)) if values else None
    with (out_dir / "chess_online_results.json").open("w", encoding="utf-8") as stream:
        json.dump({"overall": overall, "sequences": outputs}, stream, ensure_ascii=False, indent=2)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing-results-root", type=Path,
        default=Path(
            "F:/MIS_Pose_Track_Re3D/BenchmarkSuite/Results/Chessboard/"
            "Exp_Results_Chessboard_Final_Optimize_v3"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "online_chess")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--proc-scale", type=float, default=1.0)
    options = parser.parse_args()
    _install_factory()
    requested = {item.strip() for item in options.sequences.split(",") if item.strip()}
    units = []
    for rows_csv in sorted(options.existing_results_root.rglob("ours_scheme_c_nano_rows.csv")):
        relative = rows_csv.parent.relative_to(options.existing_results_root)
        sequence_name = relative.parts[0]
        if requested and sequence_name not in requested:
            continue
        with rows_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            first = next(csv.DictReader(stream), None)
        if first and first.get("image_path"):
            units.append((relative, Path(first["image_path"]).parent, rows_csv.parent / "gt"))
    if options.max_sequences > 0:
        keep = sorted({unit[0].parts[0] for unit in units})[:options.max_sequences]
        units = [unit for unit in units if unit[0].parts[0] in keep]
    outputs = []
    for relative, frames_dir, gt_dir in units:
        if not gt_dir.exists():
            print(f"[SKIP] missing GT cache: {gt_dir}")
            continue
        run_args = SimpleNamespace(
            frames_dir=str(frames_dir), out_dir=str(options.out_dir / relative),
            board_cols=0, board_rows=0, square_size_mm=0.0,
            auto_board_types="A,B,C", split_by_board_type=1, use_sb=1,
            trackers=TRACKER_KEY, precision_thresh_px=20.0,
            success_iou_thresh=0.5, gt_mask_dir="",
            reuse_gt_dir=str(gt_dir), pred_dir="", external_pred_root="",
            external_pred_map="", continue_on_error=0, profile="cascade",
            proc_scale=options.proc_scale, print_every=20, save_vis=0,
            vis_stride=10, max_frames=options.max_frames,
            save_uvh_render_debug=0,
        )
        output = benchmark.run_compare(run_args)
        output["online_unit"] = relative.as_posix()
        outputs.append(output)
    _collect(outputs, options.out_dir)


if __name__ == "__main__":
    main()

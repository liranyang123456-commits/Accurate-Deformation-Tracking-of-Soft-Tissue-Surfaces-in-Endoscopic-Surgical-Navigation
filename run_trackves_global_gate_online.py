"""Run cross-fitted global visual gates online on all TrackVes sequences."""
from __future__ import annotations

import argparse
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
import run_trackves_online as common
from global_gate_online_tracker import GlobalGateOnlineTracker

TRACKER_KEY = "ours_global_visual_gate_oof"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(benchmark.TRACKVES_ROOT))
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "online_trackves_global_gate")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    candidate = np.load(HERE / "artifacts" / "candidate_dataset.npz", allow_pickle=False)
    frame_indices = {
        name: candidate["frame_idx"][
            (candidate["domain"] == "trackves") & (candidate["sequence"] == name)
        ].astype(int)
        for name in np.unique(candidate["sequence"][candidate["domain"] == "trackves"])
    }
    original_factory = benchmark._make_tracker

    def factory(name, init_image, init_bbox, init_mask, eval_args):
        if name == TRACKER_KEY:
            sequence = str(eval_args.current_sequence)
            return GlobalGateOnlineTracker(
                init_image, init_bbox,
                HERE / "artifacts" / "global_visual_gate_folds" / f"{sequence}.pt",
                frame_indices=frame_indices[sequence],
            )
        return original_factory(name, init_image, init_bbox, init_mask, eval_args)

    benchmark._make_tracker = factory
    benchmark.TRACKER_LABELS[TRACKER_KEY] = "Ours Global Visual Gate (LOSO)"
    common.TRACKER_KEY = TRACKER_KEY
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip() for item in args.sequences.split(",") if item.strip()}
    sequences = [
        path for path in sorted(args.data_root.iterdir())
        if path.is_dir() and (path / "GT.xml").exists()
        and (not requested or path.name in requested)
    ]
    eval_args = SimpleNamespace(
        proc_scale=1.0, save_vis=False, sequential=False, print_every=20,
        save_intermediate=False, save_intermediate_overlay=False,
        force=args.force, current_sequence="",
    )
    summaries = []
    for sequence in sequences:
        eval_args.current_sequence = sequence.name
        summaries.extend(
            benchmark.evaluate_sequence(
                sequence, [TRACKER_KEY], args.out_dir, eval_args
            )
        )
    common._aggregate(summaries, args.out_dir)


if __name__ == "__main__":
    main()

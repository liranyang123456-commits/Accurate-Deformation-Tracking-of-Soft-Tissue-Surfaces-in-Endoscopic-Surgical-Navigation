"""Evaluate the non-learned conservative Scheme-C tracker on TrackVes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trackves_eval_all_methods as benchmark
import run_trackves_online as common
from conservative_scheme_c_tracker import ConservativeSchemeCTracker

TRACKER_KEY = "ours_scheme_c_conservative"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(benchmark.TRACKVES_ROOT))
    parser.add_argument("--out-dir", type=Path, default=HERE / "results" / "online_trackves_conservative")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    benchmark.apply_eval_profile("cascade")
    # Keep the frozen candidate generator on its lightweight Nano/size path.
    # The wrapper performs the conservative arbitration and global recovery.
    os.environ["MIS_NANO_FAST_SIZE_BYPASS"] = "1"
    os.environ["MIS_NANO_FAST_SIZE_NS"] = "0.0"
    os.environ["MIS_NANO_FAST_SIZE_HIST"] = "0.0"
    os.environ["MIS_NANO_FAST_SIZE_SZ"] = "0.0"
    os.environ["MIS_NANO_FAST_SIZE_IOU"] = "0.0"
    original_factory = benchmark._make_tracker

    def factory(name, init_image, init_bbox, init_mask, eval_args):
        if name == TRACKER_KEY:
            return ConservativeSchemeCTracker(init_image, init_bbox)
        return original_factory(name, init_image, init_bbox, init_mask, eval_args)

    benchmark._make_tracker = factory
    benchmark.TRACKER_LABELS[TRACKER_KEY] = "Ours Scheme-C Conservative"
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
        force=args.force,
    )
    summaries = []
    for sequence in sequences:
        summaries.extend(
            benchmark.evaluate_sequence(
                sequence, [TRACKER_KEY], args.out_dir, eval_args
            )
        )
    common._aggregate(summaries, args.out_dir)


if __name__ == "__main__":
    main()

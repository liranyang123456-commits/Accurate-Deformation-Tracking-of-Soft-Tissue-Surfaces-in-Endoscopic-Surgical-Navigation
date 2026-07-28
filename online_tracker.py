"""Final online domain router used by both benchmark runners."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FinalHybridOnlineTracker:
    name = "ours_hybrid_dynamic_online"
    label = "Ours Hybrid Dynamic Online"
    backend_name = name

    def __init__(
        self, init_image: np.ndarray, init_bbox: tuple[int, int, int, int],
        init_mask: np.ndarray | None, domain: str,
        checkpoint: str | Path, mask_checkpoint: str | Path,
        config: str | Path,
    ):
        self.domain = domain
        if domain == "trackves":
            from trackves_eval_all_methods import NanoTrackTracker
            self.tracker = NanoTrackTracker(init_image, init_bbox)
        else:
            from visual_infer_wrapper import HybridVisualMaskRouterTracker
            self.tracker = HybridVisualMaskRouterTracker(
                init_image, list(init_bbox), checkpoint=checkpoint,
                mask_checkpoint=mask_checkpoint, domain="chess", config=config,
            )

    def update(self, frame: np.ndarray) -> dict[str, Any]:
        output = self.tracker.update(frame)
        output["online_route"] = (
            "nanotrack" if self.domain == "trackves"
            else "dynamic_visual_bbox_and_mask"
        )
        return output

    def set_frame_context(self, frame_idx: int, image_path: str) -> None:
        target = getattr(self.tracker, "base", self.tracker)
        if hasattr(target, "set_frame_context"):
            target.set_frame_context(frame_idx, image_path)

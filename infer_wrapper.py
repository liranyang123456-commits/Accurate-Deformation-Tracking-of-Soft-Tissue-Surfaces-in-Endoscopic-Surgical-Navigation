"""Read-only wrapper around the frozen Scheme-C tracker plus learned gate."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from candidate_exporter import _assemble, _box, _candidate_scores, _safe, box_iou
from models import make_model


class LearnedFusionTracker:
    """Wrap, never modify, ``OursSchemeCNanoPriorTracker``.

    The frozen tracker still produces Nano/HistAB/ContourSim candidates. This
    wrapper only replaces the returned final box with a checkpointed convex
    candidate combination.
    """

    name = "ours_scheme_c_learned_fusion"

    def __init__(
        self, init_image: np.ndarray, init_bbox: list[float],
        checkpoint: str | Path, domain: str = "trackves",
        config: str | Path = "config/default.yaml",
    ):
        frozen = importlib.import_module("2026-06-03-ours_scheme_c_nano跟踪器方法")
        self.base = frozen.OursSchemeCNanoPriorTracker(init_image, init_bbox)
        self.previous = np.asarray(init_bbox, dtype=np.float32)
        self.domain = domain
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = make_model(payload["model_name"], payload["feature_dim"], cfg)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.residual_alpha = float(payload.get("residual_alpha", 1.0))
        self.mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.std = np.asarray(payload["feature_std"], dtype=np.float32)

    @torch.no_grad()
    def update(self, image: np.ndarray) -> dict[str, Any]:
        raw = self.base.update(image)
        debug = dict(raw.get("debug") or {})
        final = _box(debug.get("final_bbox") or raw.get("bbox"))
        nano = _box(debug.get("nano_bbox"))
        hist = _box(debug.get("mis_bbox"))
        if final is None or nano is None or hist is None:
            return raw
        final_sc = _candidate_scores(debug, "fusion_out") or {
            "score": _safe(debug.get("final_score")),
            "hist": _safe(debug.get("hist_sim")),
            "sim": _safe(debug.get("final_score")),
            "sz": _safe(debug.get("sz_sim")),
        }
        nano_sc = _candidate_scores(debug, "nano_raw") or {
            "score": _safe(debug.get("nano_score")),
            "nano": _safe(debug.get("nano_score")),
        }
        hist_sc = {
            "score": _safe(debug.get("final_score")),
            "hist": _safe(debug.get("hist_sim")),
            "sim": _safe(debug.get("final_score")),
            "sz": _safe(debug.get("sz_sim")),
        }
        globals_ = {
            "risk": _safe(debug.get("risk_ema", debug.get("risk_score"))),
            "pseudo3d": _safe(debug.get(
                "pseudo3d_score", debug.get("uvh_score", debug.get("render_score"))
            )),
            "reliability": float(np.mean([
                _safe(debug.get("rel_hist")), _safe(debug.get("rel_sz")),
                _safe(debug.get("rel_nano")),
            ])),
            "iou_nano_mis": _safe(debug.get("iou_mn"), box_iou(nano, hist)),
            "motion": float(np.linalg.norm(
                (nano[:2] + nano[2:] / 2) -
                (self.previous[:2] + self.previous[2:] / 2)
            ) / max(np.hypot(self.previous[2], self.previous[3]), 1.0)),
        }
        sample = _assemble(
            domain=self.domain, sequence="inference", unit="inference",
            frame_idx=0, source_path="", final=final, nano=nano, hist=hist,
            previous=self.previous, gt=final, final_scores=final_sc,
            nano_scores=nano_sc, hist_scores=hist_sc, global_values=globals_,
        )
        features = (sample.features - self.mean[None, :]) / self.std[None, :]
        pred, weights = self.model(
            torch.from_numpy(features[None]).float(),
            torch.from_numpy(sample.boxes[None]).float(),
        )
        learned_box = (
            final + self.residual_alpha * (pred[0].numpy() - final)
        ).astype(np.float32)
        self.previous = learned_box.copy()
        out = dict(raw)
        out["bbox"] = learned_box.tolist()
        out["score"] = float(np.max(weights[0].numpy()))
        out["learned_fusion"] = {
            "weights": weights[0].numpy().tolist(),
            "base_bbox": final.tolist(),
            "checkpoint_model": self.model.__class__.__name__,
            "residual_alpha": self.residual_alpha,
        }
        return out

"""Deployable frozen-tracker wrapper for visual quality/residual fusion."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from candidate_exporter import _assemble, _box, _candidate_scores, _safe, box_iou
from visual_feature_exporter import _crop, _responses
from visual_feature_exporter_dynamic import responses_dynamic
from visual_models import make_visual_model


class VisualLearnedFusionTracker:
    name = "ours_scheme_c_visual_quality_residual"

    def __init__(
        self, init_image: np.ndarray, init_bbox: list[float],
        checkpoint: str | Path, domain: str,
        config: str | Path = "config/default.yaml",
    ):
        frozen = importlib.import_module("2026-06-03-ours_scheme_c_nano跟踪器方法")
        self.base = frozen.OursSchemeCNanoPriorTracker(init_image, init_bbox)
        self.previous = np.asarray(init_bbox, dtype=np.float32)
        self.template = _crop(init_image, self.previous, context=1.0)
        self.dynamic_template = self.template.copy()
        self.domain = domain
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.visual_channels = int(payload.get("visual_channels", 4))
        self.model = make_visual_model(
            payload["feature_dim"], cfg,
            visual_channels=self.visual_channels,
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.alpha = float(payload.get("residual_alpha", 1.0))

    @torch.no_grad()
    def update(self, image: np.ndarray) -> dict[str, Any]:
        raw = self.base.update(image)
        debug = dict(raw.get("debug") or {})
        final = _box(debug.get("final_bbox") or raw.get("bbox"))
        nano, hist = _box(debug.get("nano_bbox")), _box(debug.get("mis_bbox"))
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
                (nano[:2] + nano[2:] / 2)
                - (self.previous[:2] + self.previous[2:] / 2)
            ) / max(np.hypot(self.previous[2], self.previous[3]), 1.0)),
        }
        sample = _assemble(
            domain=self.domain, sequence="inference", unit="inference",
            frame_idx=0, source_path="", final=final, nano=nano, hist=hist,
            previous=self.previous, gt=final, final_scores=final_sc,
            nano_scores=nano_sc, hist_scores=hist_sc, global_values=globals_,
        )
        features = (sample.features - self.mean[None, :]) / self.std[None, :]
        if self.visual_channels == 6:
            maps = np.stack([
                responses_dynamic(
                    _crop(image, box), self.template, self.dynamic_template
                )
                for box in sample.boxes
            ])
        else:
            maps = np.stack([
                _responses(_crop(image, box), self.template) for box in sample.boxes
            ])
        maps = maps.astype(np.float32) / 255.0
        self._last_response_maps = maps.copy()
        self._last_base_bbox = final.copy()
        self._last_image_shape = image.shape[:2]
        details = self.model.forward_details(
            torch.from_numpy(features[None]).float(),
            torch.from_numpy(sample.boxes[None]).float(),
            torch.from_numpy(maps[None]).float(),
        )
        raw_visual = details["pred"][0].numpy()
        pred = (final + self.alpha * (raw_visual - final)).astype(np.float32)
        self.previous = pred.copy()
        if globals_["reliability"] >= 0.55 and globals_["risk"] <= 0.50:
            import cv2
            current = _crop(image, final, context=1.0)
            self.dynamic_template = cv2.addWeighted(
                self.dynamic_template, 0.90, current, 0.10, 0.0
            )
        out = dict(raw)
        out["bbox"] = pred.tolist()
        out["visual_learned_fusion"] = {
            "weights": details["weights"][0].numpy().tolist(),
            "predicted_quality": details["quality"][0].numpy().tolist(),
            "residual_alpha": self.alpha,
        }
        return out


class HybridVisualRouterTracker(VisualLearnedFusionTracker):
    """Final policy: visual residual on chess, frozen Nano box on TrackVes."""

    name = "ours_hybrid_visual_router"

    @torch.no_grad()
    def update(self, image: np.ndarray) -> dict[str, Any]:
        out = super().update(image)
        if self.domain == "trackves":
            debug = dict(out.get("debug") or {})
            nano = _box(debug.get("nano_bbox"))
            if nano is not None:
                out["bbox"] = nano.tolist()
                out["hybrid_route"] = "nano"
                self.previous = nano.copy()
        else:
            out["hybrid_route"] = "visual_quality_residual"
        return out


class HybridVisualMaskRouterTracker(HybridVisualRouterTracker):
    """BBox SOTA router plus chess-only real-mask head."""

    name = "ours_hybrid_dynamic_bbox_mask_router"

    def __init__(self, *args, mask_checkpoint: str | Path, **kwargs):
        super().__init__(*args, **kwargs)
        from run_mask_oof import LightMaskHead
        payload = torch.load(mask_checkpoint, map_location="cpu", weights_only=False)
        self.mask_head = LightMaskHead(int(payload["channels"]))
        self.mask_head.load_state_dict(payload["state_dict"])
        self.mask_head.eval()
        self.mask_threshold = float(payload["threshold"])
        self.mask_context = float(payload.get("roi_context", 1.5))

    @torch.no_grad()
    def update(self, image: np.ndarray) -> dict[str, Any]:
        out = super().update(image)
        h, w = image.shape[:2]
        if self.domain == "chess" and hasattr(self, "_last_response_maps"):
            probability = torch.sigmoid(self.mask_head(
                torch.from_numpy(self._last_response_maps[0:1]).float()
            ))[0].numpy()
            roi_mask = (probability >= self.mask_threshold).astype(np.uint8) * 255
            x, y, bw, bh = map(float, self._last_base_bbox)
            cw, ch = max(4.0, bw * self.mask_context), max(4.0, bh * self.mask_context)
            x1, y1 = max(0, int(round(x + bw / 2 - cw / 2))), max(0, int(round(y + bh / 2 - ch / 2)))
            x2, y2 = min(w, int(round(x + bw / 2 + cw / 2))), min(h, int(round(y + bh / 2 + ch / 2)))
            full = np.zeros((h, w), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                import cv2
                full[y1:y2, x1:x2] = cv2.resize(
                    roi_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
                )
            out["mask_full"] = full
            out["mask_route"] = "learned_chess_mask"
        else:
            full = np.zeros((h, w), dtype=np.uint8)
            x, y, bw, bh = map(int, out["bbox"])
            full[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = 255
            out["mask_full"] = full
            out["mask_route"] = "bbox_mask"
        return out

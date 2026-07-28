"""Non-learned conservative Scheme-C tracker with Nano-safe arbitration."""
from __future__ import annotations

import importlib
from typing import Any

import cv2
import numpy as np

from candidate_exporter import _safe, box_iou


def _crop_raw(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = map(int, np.round(box))
    patch = image[
        max(0, y):min(image.shape[0], y + h),
        max(0, x):min(image.shape[1], x + w),
    ]
    if patch.size == 0:
        raise ValueError("Initial template crop is empty")
    return patch.copy()


def _gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


def _clip(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = map(float, box)
    bw, bh = min(max(4.0, bw), w), min(max(4.0, bh), h)
    return np.asarray([
        min(max(0.0, x), max(0.0, w - bw)),
        min(max(0.0, y), max(0.0, h - bh)), bw, bh,
    ], dtype=np.float32)


def global_template_psr(
    image: np.ndarray, template: np.ndarray,
    scales: tuple[float, ...] = (0.70, 0.85, 1.0, 1.15, 1.30),
    work_scale: float = 0.5,
) -> tuple[np.ndarray, float, float, float]:
    frame = cv2.resize(image, None, fx=work_scale, fy=work_scale)
    frame_gray, template_gray = _gray(frame), _gray(template)
    best = (np.zeros(4, dtype=np.float32), -1.0, 0.0, 0.0)
    for scale in scales:
        tw = max(8, int(round(template.shape[1] * scale * work_scale)))
        th = max(8, int(round(template.shape[0] * scale * work_scale)))
        if tw >= frame_gray.shape[1] or th >= frame_gray.shape[0]:
            continue
        resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, peak, _, location = cv2.minMaxLoc(response)
        mean, std = float(response.mean()), float(response.std())
        psr = float((peak - mean) / max(std, 1e-6))
        suppressed = response.copy()
        radius_x, radius_y = max(2, tw // 2), max(2, th // 2)
        x1, y1 = max(0, location[0] - radius_x), max(0, location[1] - radius_y)
        x2 = min(suppressed.shape[1], location[0] + radius_x + 1)
        y2 = min(suppressed.shape[0], location[1] + radius_y + 1)
        suppressed[y1:y2, x1:x2] = -1.0
        second = float(suppressed.max()) if suppressed.size else -1.0
        margin = float(peak - second)
        if peak > best[1]:
            box = np.asarray([
                location[0] / work_scale, location[1] / work_scale,
                tw / work_scale, th / work_scale,
            ], dtype=np.float32)
            best = (_clip(box, image.shape[:2]), float(peak), psr, margin)
    return best


class ConservativeSchemeCTracker:
    name = "ours_scheme_c_conservative"
    label = "Ours Scheme-C Conservative"
    backend_name = name

    def __init__(self, init_image: np.ndarray, init_bbox: tuple[int, int, int, int]):
        frozen = importlib.import_module("2026-06-03-ours_scheme_c_nano跟踪器方法")
        from trackves_eval_all_methods import NanoTrackTracker
        # Frozen Scheme-C is initialized only as an inference-safe HistAB /
        # ContourSimilarityNet verifier. Its expensive search/update is not run.
        self.verifier = frozen.OursSchemeCNanoPriorTracker(init_image, init_bbox)
        self.nano = NanoTrackTracker(init_image, init_bbox)
        self.template = _crop_raw(init_image, np.asarray(init_bbox, dtype=np.float32))
        self.init_box = np.asarray(init_bbox, dtype=np.float32)
        self.previous_nano = self.init_box.copy()
        self.bad_streak = 0
        self.update_count = 0
        self.route_counts = {"nano": 0, "size": 0, "hist": 0, "global": 0}

    def propagate(self, frame: np.ndarray) -> None:
        self.nano.update(frame)

    def _verify(
        self, image: np.ndarray, boxes: list[np.ndarray],
    ) -> list[dict[str, float]]:
        tuples = [tuple(map(int, np.round(box))) for box in boxes]
        th68 = self.verifier._defm._apply_th68(image)
        contours = self.verifier._contour_similarity_for_bboxes(th68, tuples)
        output = []
        for box, contour in zip(tuples, contours):
            hist = float(np.clip(self.verifier._quick_hist_sim(image, box), 0.0, 1.0))
            size = float(np.clip(self.verifier._quick_sz_sim(image, box), 0.0, 1.0))
            quality = 0.38 * hist + 0.38 * float(contour) + 0.24 * size
            output.append({
                "hist": hist, "contour": float(contour),
                "size": size, "quality": quality,
            })
        return output

    @staticmethod
    def _size_candidates(nano: np.ndarray) -> list[np.ndarray]:
        cx, cy = nano[0] + nano[2] / 2, nano[1] + nano[3] / 2
        candidates = []
        for sw in (0.90, 1.0, 1.10):
            for sh in (0.90, 1.0, 1.10):
                box = nano.copy()
                box[2], box[3] = nano[2] * sw, nano[3] * sh
                box[0], box[1] = cx - box[2] / 2, cy - box[3] / 2
                candidates.append(box)
        return candidates

    def update(self, image: np.ndarray) -> dict[str, Any]:
        raw = self.nano.update(image)
        nano = np.asarray(raw["bbox"], dtype=np.float32)
        nano_score = _safe(raw.get("score"))
        size_candidates = self._size_candidates(nano)
        verification = self._verify(image, size_candidates)
        best_index = int(np.argmax([item["quality"] for item in verification]))
        best_scores = verification[best_index]
        strict = (
            best_scores["hist"] >= 0.82
            and best_scores["contour"] >= 0.80
            and best_scores["size"] >= 0.72
        )
        candidate = size_candidates[best_index].copy() if strict else nano.copy()
        route = "size" if strict and best_index != 4 else "nano"
        motion = float(np.linalg.norm(
            (nano[:2] + nano[2:] / 2)
            - (self.previous_nano[:2] + self.previous_nano[2:] / 2)
        ) / max(np.hypot(self.previous_nano[2], self.previous_nano[3]), 1.0))
        area_ratio = float(
            (nano[2] * nano[3]) / max(self.init_box[2] * self.init_box[3], 1.0)
        )
        suspect = (
            nano_score < 0.40 or motion > 1.0 or area_ratio < 0.35
            or area_ratio > 2.8
        )
        self.bad_streak = self.bad_streak + 1 if suspect else max(0, self.bad_streak - 1)
        self.update_count += 1
        global_info: dict[str, float] = {}
        if self.bad_streak >= 3 or self.update_count % 10 == 0:
            global_box, peak, psr, margin = global_template_psr(image, self.template)
            global_scores = self._verify(image, [global_box])[0]
            global_info = {
                "peak": peak, "psr": psr, "margin": margin, **global_scores,
            }
            global_agreement = box_iou(nano, global_box)
            accept = (
                peak >= 0.62 and psr >= 7.5 and margin >= 0.035
                and global_scores["hist"] >= 0.82
                and global_scores["contour"] >= 0.80
                and global_scores["size"] >= 0.72
                and (
                    global_agreement >= 0.15
                    or (psr >= 10.0 and margin >= 0.060)
                )
            )
            if accept:
                candidate, route = global_box, "global"
                self.bad_streak = 0
        candidate = _clip(candidate, image.shape[:2])
        self.previous_nano = nano.copy()
        self.route_counts[route] += 1
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = map(int, np.round(candidate))
        mask[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = 255
        output = dict(raw)
        output.update({
            "bbox": candidate.tolist(), "mask_full": mask,
            "conservative_route": route,
            "conservative_debug": {
                "bad_streak": self.bad_streak, "motion": motion,
                "area_ratio": area_ratio, "size_scores": best_scores,
                "global": global_info, "route_counts": dict(self.route_counts),
            },
        })
        return output

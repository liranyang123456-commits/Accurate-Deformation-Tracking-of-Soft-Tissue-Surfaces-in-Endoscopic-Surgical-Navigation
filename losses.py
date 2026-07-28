"""Differentiable box losses for convex candidate fusion."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [box[..., 0], box[..., 1], box[..., 0] + box[..., 2],
         box[..., 1] + box[..., 3]],
        dim=-1,
    )


def aligned_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(dim=-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(dim=-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def generalized_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, bx = xywh_to_xyxy(a), xywh_to_xyxy(b)
    iou = aligned_iou(a, b)
    c_lt = torch.minimum(ax[..., :2], bx[..., :2])
    c_rb = torch.maximum(ax[..., 2:], bx[..., 2:])
    c_area = (c_rb - c_lt).clamp(min=0).prod(dim=-1).clamp(min=1e-6)
    lt = torch.maximum(ax[..., :2], bx[..., :2])
    rb = torch.minimum(ax[..., 2:], bx[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (ax[..., 2:] - ax[..., :2]).clamp(min=0).prod(dim=-1)
    area_b = (bx[..., 2:] - bx[..., :2]).clamp(min=0).prod(dim=-1)
    union = area_a + area_b - inter
    return iou - (c_area - union) / c_area


def fusion_loss(
    pred: torch.Tensor,
    weights: torch.Tensor,
    candidate_boxes: torch.Tensor,
    gt: torch.Tensor,
    oracle_ce_weight: float,
    temporal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    giou = generalized_iou(pred, gt)
    candidate_iou = aligned_iou(candidate_boxes, gt[:, None, :])
    oracle = candidate_iou.argmax(dim=1)
    ce = F.nll_loss(torch.log(weights.clamp(min=1e-7)), oracle)
    # Candidate 3 is previous frame. Normalize pixels by its scale.
    previous = candidate_boxes[:, 3]
    scale = torch.stack(
        [previous[:, 2], previous[:, 3], previous[:, 2], previous[:, 3]], dim=1
    ).clamp(min=1.0)
    temporal = ((pred - previous).abs() / scale).mean()
    loss = (1.0 - giou).mean() + oracle_ce_weight * ce + temporal_weight * temporal
    return loss, {
        "giou": float(giou.mean().detach()),
        "oracle_ce": float(ce.detach()),
        "temporal": float(temporal.detach()),
    }


def visual_fusion_loss(
    details: dict[str, torch.Tensor],
    candidate_boxes: torch.Tensor,
    gt: torch.Tensor,
    *,
    residual_limit: float,
    quality_weight: float,
    residual_weight: float,
    temporal_weight: float,
    domain_trackves: torch.Tensor | None = None,
    trackves_size_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = details["pred"]
    giou = generalized_iou(pred, gt)
    quality_target = aligned_iou(candidate_boxes, gt[:, None, :]).detach()
    quality_loss = F.smooth_l1_loss(details["quality"], quality_target)
    base = candidate_boxes[:, 0]
    previous = candidate_boxes[:, 3]
    limit = max(float(residual_limit), 1e-4)
    target_residual = torch.stack(
        [
            (gt[:, 0] - base[:, 0]) / previous[:, 2].clamp(min=1.0) / limit,
            (gt[:, 1] - base[:, 1]) / previous[:, 3].clamp(min=1.0) / limit,
            torch.log(gt[:, 2].clamp(min=1.0) / base[:, 2].clamp(min=1.0)) / limit,
            torch.log(gt[:, 3].clamp(min=1.0) / base[:, 3].clamp(min=1.0)) / limit,
        ],
        dim=1,
    ).clamp(-1.0, 1.0)
    residual_loss = F.smooth_l1_loss(details["residual_unit"], target_residual)
    size_error = torch.abs(
        torch.log(pred[:, 2:].clamp(min=1.0) / gt[:, 2:].clamp(min=1.0))
    ).mean(dim=1)
    if domain_trackves is None:
        size_loss = size_error.mean()
    else:
        size_loss = (
            size_error * (1.0 + domain_trackves.float() * trackves_size_weight)
        ).mean()
    scale = torch.stack(
        [previous[:, 2], previous[:, 3], previous[:, 2], previous[:, 3]], dim=1
    ).clamp(min=1.0)
    temporal = ((pred - previous).abs() / scale).mean()
    total = (
        (1.0 - giou).mean()
        + quality_weight * quality_loss
        + residual_weight * residual_loss
        + trackves_size_weight * size_loss
        + temporal_weight * temporal
    )
    return total, {
        "giou": float(giou.mean().detach()),
        "quality": float(quality_loss.detach()),
        "residual": float(residual_loss.detach()),
        "size": float(size_loss.detach()),
        "temporal": float(temporal.detach()),
    }

"""Create compact OOF comparison and per-sequence delta figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", default="artifacts/evaluation")
    ap.add_argument("--predictions", default="artifacts/oof_predictions.csv")
    args = ap.parse_args()
    evaluation = Path(args.evaluation)
    summary = pd.read_csv(evaluation / "summary_metrics.csv")
    sequence = pd.read_csv(evaluation / "sequence_metrics.csv")
    selection = json.loads((evaluation / "selection.json").read_text(encoding="utf-8"))
    best = selection["best_learned"]
    models = [
        "nano", "hist", "full_fusion", "domain_router", "linear",
        "tiny_mlp", "micro_transformer", "oracle",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = [
        "#8da0cb", "#66c2a5", "#4c78a8", "#2a9d8f",
        "#f2cf5b", "#f58518", "#e45756", "#777777",
    ]
    for ax, domain in zip(axes, ["trackves", "chess"]):
        part = summary[summary.domain == domain].set_index("model").reindex(models)
        y = part["bbox_iou"].to_numpy()
        err = part["bbox_iou_std"].fillna(0).to_numpy()
        ax.bar(np.arange(len(models)), y, yerr=err, capsize=2, color=colors)
        ax.set_xticks(np.arange(len(models)), [m.replace("_", "\n") for m in models], rotation=25)
        ax.set_ylabel("Sequence-mean BBox-IoU")
        ax.set_title(domain.capitalize() + " OOF")
        ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(evaluation / f"FigFusionOOF.{suffix}", dpi=220)
    plt.close(fig)
    pivot = sequence.pivot_table(
        index=["domain", "sequence"], columns="model", values="bbox_iou"
    )
    delta = (pivot[best] - pivot["full_fusion"]).sort_index()
    labels = [f"{d}:{s}" for d, s in delta.index]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    color = np.where(delta.to_numpy() >= 0, "#2a9d8f", "#e76f51")
    ax.bar(np.arange(len(delta)), delta.to_numpy(), color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(delta)), labels, rotation=60, ha="right")
    ax.set_ylabel(f"{best} minus Full fusion BBox-IoU")
    ax.set_title("Per-sequence OOF improvement")
    ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(evaluation / f"FigFusionSequenceDelta.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

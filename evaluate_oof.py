"""Summarize OOF predictions, significance, speed, ablations, and LaTeX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MODEL_ORDER = [
    "nano", "hist", "full_fusion", "domain_router", "heuristic_score", "linear",
    "tiny_mlp", "micro_transformer", "oracle",
]
LEARNED = ["linear", "tiny_mlp", "micro_transformer"]


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values))
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="artifacts/oof_predictions.csv")
    ap.add_argument("--manifest", default="artifacts/run_manifest.json")
    ap.add_argument("--out-dir", default="artifacts/evaluation")
    args = ap.parse_args()
    pred_path, manifest_path = Path(args.predictions), Path(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_path)
    seq = (
        df.groupby(["domain", "sequence", "model"], as_index=False)
        .agg(
            bbox_iou=("bbox_iou", "mean"),
            success_05=("success_05", "mean"),
            precision_20=("precision_20", "mean"),
            center_error_px=("center_error_px", "mean"),
            frames=("frame_idx", "count"),
        )
    )
    summary = (
        seq.groupby(["domain", "model"], as_index=False)
        .agg(
            bbox_iou=("bbox_iou", "mean"),
            bbox_iou_std=("bbox_iou", "std"),
            success_05=("success_05", "mean"),
            precision_20=("precision_20", "mean"),
            center_error_px=("center_error_px", "mean"),
            sequences=("sequence", "count"),
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    speed_rows = []
    for fold in manifest["training"]:
        for model in fold["models"]:
            speed_rows.append({
                "model": model["model"], "fold_id": model["fold_id"],
                "parameters": model["parameters"],
                "ms_per_frame": model["ms_per_frame"],
            })
    speed = pd.DataFrame(speed_rows)
    speed_summary = (
        speed.groupby("model", as_index=False)
        .agg(parameters=("parameters", "first"), ms_per_frame=("ms_per_frame", "mean"))
    )
    summary = summary.merge(speed_summary, on="model", how="left")
    summary["fps_fuser_only"] = 1000.0 / summary["ms_per_frame"]
    seq.to_csv(out / "sequence_metrics.csv", index=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    # Sequence-level paired tests avoid treating adjacent frames as independent.
    tests = []
    for domain in sorted(seq.domain.unique()):
        pivot = seq[seq.domain == domain].pivot(index="sequence", columns="model", values="bbox_iou")
        for model in ["domain_router", "heuristic_score", *LEARNED]:
            delta = pivot[model] - pivot["full_fusion"]
            try:
                result = wilcoxon(delta, alternative="greater", zero_method="zsplit")
                p = float(result.pvalue)
                statistic = float(result.statistic)
            except ValueError:
                p, statistic = 1.0, 0.0
            tests.append({
                "domain": domain, "model": model, "baseline": "full_fusion",
                "mean_delta_iou": float(delta.mean()),
                "median_delta_iou": float(delta.median()),
                "positive_sequences": int((delta > 0).sum()),
                "n_sequences": int(delta.notna().sum()),
                "wilcoxon_statistic": statistic, "p_one_sided": p,
            })
    adjusted = _holm([x["p_one_sided"] for x in tests])
    for row, p_adj in zip(tests, adjusted):
        row["p_holm"] = p_adj
        row["significant_005"] = bool(p_adj < 0.05 and row["mean_delta_iou"] > 0)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out / "wilcoxon_vs_full_fusion.csv", index=False)
    # Equal weight to the two domains, then validation-only protocol determines
    # checkpoints. Oracle is excluded from deployable selection.
    deployable = ["full_fusion", "domain_router", "heuristic_score", *LEARNED]
    score = (
        summary[summary.model.isin(deployable)]
        .pivot(index="model", columns="domain", values="bbox_iou")
    )
    score["equal_domain_iou"] = score.mean(axis=1)
    best_model = str(score["equal_domain_iou"].idxmax())
    best_learned = str(score.loc[LEARNED, "equal_domain_iou"].idxmax())
    learned_beats_full = bool(
        score.loc[best_learned, "equal_domain_iou"]
        > score.loc["full_fusion", "equal_domain_iou"]
    )
    significant_both = bool(
        tests_df[(tests_df.model == best_learned)].significant_005.all()
    )
    selection = {
        "selected_deployable": best_model,
        "best_learned": best_learned,
        "learned_beats_full_equal_domain": learned_beats_full,
        "learned_significant_in_both_domains_after_holm": significant_both,
        "may_name_learned_fusion_winner": learned_beats_full and significant_both,
        "equal_domain_scores": score["equal_domain_iou"].to_dict(),
        "selection_rule": (
            "Highest equal-domain sequence-mean BBox-IoU. A learned method is "
            "called the winner only if it improves this score and is significant "
            "in both domains after Holm correction."
        ),
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    # Compact paper-ready table. Cov/Mask-IoU is deliberately omitted because
    # candidate-box mixing does not generate a new segmentation mask.
    tex_rows = []
    for _, row in summary.sort_values(
        ["domain", "model"], key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)})
        if s.name == "model" else s
    ).iterrows():
        method_tex = str(row["model"]).replace("_", r"\_")
        tex_rows.append(
            f"{row['domain']} & {method_tex} & "
            f"{row['bbox_iou']:.3f} & {row['success_05']:.3f} & "
            f"{row['precision_20']:.3f} & {row['center_error_px']:.2f} \\\\"
        )
    tex = "\n".join([
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Domain & Method & BBox-IoU & Success@0.5 & Precision@20 & CE (px) \\",
        r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}",
    ])
    (out / "table_oof.tex").write_text(tex, encoding="utf-8")
    report_lines = [
        "# Nano–HistAB–ContourSim learned fusion: OOF results",
        "",
        "This report contains only sequence-level out-of-fold results. Ground truth "
        "was used for training loss and offline metrics, never as an inference feature.",
        "",
        "## Model selection",
        "",
        f"- Selected deployable method: `{best_model}`",
        f"- Best learned model: `{best_learned}`",
        f"- Learned model improves equal-domain BBox-IoU: `{learned_beats_full}`",
        f"- Significant in both domains after Holm correction: `{significant_both}`",
        f"- Eligible to be named the learned-fusion winner: "
        f"`{selection['may_name_learned_fusion_winner']}`",
        "",
        "## OOF summary",
        "",
        "| Domain | Method | BBox-IoU | Success@0.5 | Precision@20 | CE (px) | Params | Fuser ms/frame |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["domain", "bbox_iou"], ascending=[True, False]).iterrows():
        params = "-" if pd.isna(row["parameters"]) else str(int(row["parameters"]))
        latency = "-" if pd.isna(row["ms_per_frame"]) else f"{row['ms_per_frame']:.4f}"
        report_lines.append(
            f"| {row['domain']} | {row['model']} | {row['bbox_iou']:.4f} | "
            f"{row['success_05']:.4f} | {row['precision_20']:.4f} | "
            f"{row['center_error_px']:.2f} | {params} | {latency} |"
        )
    report_lines.extend([
        "",
        "## Statistical and scope notes",
        "",
        "- Wilcoxon tests use paired sequence means, not correlated frame samples.",
        "- `oracle` is a GT-only upper bound and is never deployable.",
        "- Reported latency is the fusion gate alone; candidate tracker runtime is unchanged.",
        "- A new Mask/Cov-IoU is not reported: convex box fusion does not produce a new "
        "segmentation mask. Reusing a source mask would not be a valid learned-box metric.",
        "- Full split records, checkpoint hashes, seeds, and commands are in "
        "`artifacts/run_manifest.json`.",
        "",
        "See `artifacts/evaluation/wilcoxon_vs_full_fusion.csv` for all corrected tests.",
    ])
    (out.parent.parent / "RESULTS_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

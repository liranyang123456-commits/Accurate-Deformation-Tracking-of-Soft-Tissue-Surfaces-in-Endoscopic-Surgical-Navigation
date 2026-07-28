"""Evaluate visual OOF fusion, quality calibration, and continuous oracle."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from candidate_exporter import box_iou
from dataset_builder import load_arrays
from visual_feature_exporter import _crop, _responses
from visual_feature_exporter_dynamic import responses_dynamic


def _center_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a[:2] + a[2:] / 2) - (b[:2] + b[2:] / 2)))


def _continuous_oracle(arrays: dict[str, np.ndarray], limit: float) -> pd.DataFrame:
    rows = []
    for i, (candidates, gt) in enumerate(zip(arrays["boxes"], arrays["gt"])):
        previous = candidates[3]
        corrected = []
        for box in candidates:
            dx = np.clip(gt[0] - box[0], -limit * previous[2], limit * previous[2])
            dy = np.clip(gt[1] - box[1], -limit * previous[3], limit * previous[3])
            dw = np.clip(np.log(max(gt[2], 1) / max(box[2], 1)), -limit, limit)
            dh = np.clip(np.log(max(gt[3], 1) / max(box[3], 1)), -limit, limit)
            corrected.append(np.asarray([
                box[0] + dx, box[1] + dy, box[2] * np.exp(dw), box[3] * np.exp(dh)
            ], dtype=np.float32))
        pred = max(corrected, key=lambda x: box_iou(x, gt))
        iou = box_iou(pred, gt)
        rows.append({
            "fold_id": "gt_only", "model": "continuous_oracle",
            "domain": str(arrays["domain"][i]),
            "sequence": str(arrays["sequence"][i]),
            "unit": str(arrays["unit"][i]),
            "frame_idx": int(arrays["frame_idx"][i]),
            "bbox_iou": iou, "center_error_px": _center_error(pred, gt),
            "success_05": int(iou >= 0.5),
            "precision_20": int(_center_error(pred, gt) <= 20),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="artifacts/oof_predictions.csv")
    ap.add_argument("--visual", default="artifacts/visual_oof_predictions.csv")
    ap.add_argument("--dataset", default="artifacts/candidate_dataset.npz")
    ap.add_argument("--responses", default="artifacts/visual_responses.npz")
    ap.add_argument("--manifest", default="artifacts/visual_run_manifest.json")
    ap.add_argument("--out-dir", default="artifacts/visual_evaluation")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--hybrid-name", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base)
    visual = pd.read_csv(args.visual)
    visual_name = args.model_name or str(visual.model.unique()[0])
    hybrid_name = args.hybrid_name or f"hybrid_{visual_name}"
    hybrid = pd.concat([
        visual[visual.domain == "chess"].copy(),
        base[(base.domain == "trackves") & (base.model == "nano")].copy(),
    ], ignore_index=True)
    hybrid["model"] = hybrid_name
    arrays = load_arrays(Path(args.dataset))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    limit = float(manifest["training"][0]["model"]["history"] is not None)  # manifest integrity check
    del limit
    continuous = _continuous_oracle(arrays, 0.25)
    all_predictions = pd.concat([base, visual, hybrid, continuous], ignore_index=True)
    sequence = (
        all_predictions.groupby(["domain", "sequence", "model"], as_index=False)
        .agg(
            bbox_iou=("bbox_iou", "mean"), success_05=("success_05", "mean"),
            precision_20=("precision_20", "mean"),
            center_error_px=("center_error_px", "mean"), frames=("frame_idx", "count"),
        )
    )
    summary = (
        sequence.groupby(["domain", "model"], as_index=False)
        .agg(
            bbox_iou=("bbox_iou", "mean"), bbox_iou_std=("bbox_iou", "std"),
            success_05=("success_05", "mean"), precision_20=("precision_20", "mean"),
            center_error_px=("center_error_px", "mean"), sequences=("sequence", "count"),
        )
    )
    model_meta = [x["model"] for x in manifest["training"]]
    chess_idx = int(np.flatnonzero(arrays["domain"] == "chess")[0])
    benchmark_image = cv2.imread(str(arrays["source_path"][chess_idx]), cv2.IMREAD_COLOR)
    template = _crop(benchmark_image, arrays["gt"][chess_idx], context=1.0)
    dynamic_benchmark = "dynamic" in Path(args.responses).stem
    response_description = (
        "Six static+adaptive-template response channels"
        if dynamic_benchmark else "Four static-template response channels"
    )
    start = time.perf_counter()
    repeats = 100
    for _ in range(repeats):
        for candidate in arrays["boxes"][chess_idx]:
            patch = _crop(benchmark_image, candidate)
            if dynamic_benchmark:
                responses_dynamic(patch, template, template)
            else:
                _responses(patch, template)
    preprocess_ms = 1000.0 * (time.perf_counter() - start) / repeats
    model_ms = float(np.mean([x["ms_per_frame"] for x in model_meta]))
    summary.loc[summary.model == visual_name, "parameters"] = int(
        model_meta[0]["parameters"]
    )
    summary.loc[summary.model == visual_name, "ms_per_frame"] = float(
        model_ms
    )
    summary.to_csv(out / "summary_metrics.csv", index=False)
    sequence.to_csv(out / "sequence_metrics.csv", index=False)
    tests = []
    for domain in ("trackves", "chess"):
        pivot = sequence[sequence.domain == domain].pivot(
            index="sequence", columns="model", values="bbox_iou"
        )
        for model, baseline in (
            (visual_name, "full_fusion"),
            (visual_name, "domain_router"),
            (hybrid_name, "domain_router"),
        ):
            delta = pivot[model] - pivot[baseline]
            result = wilcoxon(delta, alternative="greater", zero_method="zsplit")
            tests.append({
                "domain": domain, "model": model, "baseline": baseline,
                "mean_delta_iou": float(delta.mean()),
                "positive_sequences": int((delta > 0).sum()),
                "n_sequences": len(delta), "p_one_sided": float(result.pvalue),
            })
    tests_df = pd.DataFrame(tests)
    tests_df["p_bonferroni"] = np.minimum(1.0, tests_df.p_one_sided * len(tests_df))
    tests_df.to_csv(out / "wilcoxon_visual.csv", index=False)
    # Quality-head diagnostics over all OOF frames/candidates.
    quality = np.stack(visual.predicted_candidate_quality.map(json.loads))
    actual = np.asarray([
        [box_iou(candidate, gt) for candidate in candidates]
        for candidates, gt in zip(arrays["boxes"], arrays["gt"])
    ])
    rho = float(spearmanr(quality.ravel(), actual.ravel()).statistic)
    top1 = float(np.mean(quality.argmax(1) == actual.argmax(1)))
    equal_domain = (
        summary[summary.model.isin([
            "full_fusion", "domain_router", visual_name, hybrid_name,
            "oracle", "continuous_oracle",
        ])]
        .pivot(index="model", columns="domain", values="bbox_iou")
        .mean(axis=1)
        .to_dict()
    )
    visual_beats_router = (
        equal_domain[visual_name] > equal_domain["domain_router"]
    )
    significant_both = bool(
        (tests_df[tests_df.model == hybrid_name].p_bonferroni < 0.05).all()
        and (tests_df[tests_df.model == hybrid_name].mean_delta_iou > 0).all()
    )
    best_observed = max(
        ("domain_router", visual_name, hybrid_name),
        key=lambda name: equal_domain[name],
    )
    selection = {
        "equal_domain_scores": equal_domain,
        "quality_spearman": rho, "quality_top1_accuracy": top1,
        "visual_beats_domain_router": visual_beats_router,
        "visual_model": visual_name,
        "hybrid_model": hybrid_name,
        "chess_primary_p_one_sided": float(
            tests_df[
                (tests_df.domain == "chess")
                & (tests_df.model == visual_name)
                & (tests_df.baseline == "full_fusion")
            ].p_one_sided.iloc[0]
        ),
        "significant_both_domains": significant_both,
        "selected_deployable": best_observed,
        "selected_improvement_statistically_confirmed": significant_both,
        "visual_preprocess_ms_per_frame": preprocess_ms,
        "visual_model_ms_per_frame": model_ms,
        "visual_total_overhead_ms_per_frame": preprocess_ms + model_ms,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    methods = [
        "full_fusion", "domain_router", "linear", "tiny_mlp",
        "micro_transformer", visual_name, hybrid_name,
        "oracle", "continuous_oracle",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, domain in zip(axes, ("trackves", "chess")):
        part = summary[summary.domain == domain].set_index("model").reindex(methods)
        ax.bar(np.arange(len(methods)), part.bbox_iou, color="#4c78a8")
        ax.set_xticks(np.arange(len(methods)), [x.replace("_", "\n") for x in methods], rotation=25)
        ax.set_title(domain.capitalize() + " sequence-level OOF")
        ax.set_ylabel("BBox-IoU")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out / "FigVisualFusionOOF.png", dpi=220)
    fig.savefig(out / "FigVisualFusionOOF.pdf")
    plt.close(fig)
    tex_rows = []
    for _, r in summary[summary.model.isin(methods)].sort_values(
        ["domain", "bbox_iou"], ascending=[True, False]
    ).iterrows():
        tex_rows.append(
            f"{r.domain} & {str(r.model).replace('_', chr(92) + '_')} & "
            f"{r.bbox_iou:.3f} & {r.success_05:.3f} & "
            f"{r.precision_20:.3f} & {r.center_error_px:.2f} \\\\"
        )
    (out / "table_visual_oof.tex").write_text("\n".join([
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Domain & Method & BBox-IoU & Success@0.5 & Precision@20 & CE (px) \\",
        r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}",
    ]), encoding="utf-8")
    report = [
        "# Visual quality + continuous residual fusion",
        "",
        f"- Selected deployable: `{selection['selected_deployable']}`",
        f"- Quality Spearman correlation: {rho:.4f}",
        f"- Candidate top-1 accuracy: {top1:.4f}",
        f"- Visual preprocessing: {preprocess_ms:.3f} ms/frame",
        f"- Visual model: {model_ms:.3f} ms/frame",
        f"- Visual model beats domain router: `{visual_beats_router}`",
        f"- Selected improvement significant in both domains: `{significant_both}`",
        f"- Chess primary one-sided Wilcoxon p: "
        f"{selection['chess_primary_p_one_sided']:.4f} (exploratory after model selection)",
        f"- Chess visual BBox-IoU: "
        f"{summary[(summary.domain == 'chess') & (summary.model == visual_name)].bbox_iou.iloc[0]:.4f} "
        f"(Full fusion "
        f"{summary[(summary.domain == 'chess') & (summary.model == 'full_fusion')].bbox_iou.iloc[0]:.4f})",
        f"- TrackVes hybrid BBox-IoU: "
        f"{summary[(summary.domain == 'trackves') & (summary.model == hybrid_name)].bbox_iou.iloc[0]:.4f}",
        "",
        "## Equal-domain sequence-mean BBox-IoU",
        "",
        *[f"- `{k}`: {v:.4f}" for k, v in sorted(equal_domain.items(), key=lambda x: -x[1])],
        "",
        "The continuous oracle is GT-only and measures headroom after permitting "
        "bounded box correction outside the original convex candidate set.",
        "All reported visual-model predictions are sequence-level OOF.",
        f"{response_description} are synchronously derived from the image and "
        "initial template; they are not claimed to be internal NanoTrack response maps.",
    ]
    (out.parent.parent / "VISUAL_RESULTS_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

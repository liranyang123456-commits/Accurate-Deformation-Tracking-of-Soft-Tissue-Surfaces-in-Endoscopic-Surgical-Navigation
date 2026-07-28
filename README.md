# Nano–HistAB–ContourSim learned box fusion

This directory is a standalone experiment. It does not edit the frozen
Scheme-C/Nano, HistAB, ContourSim, benchmark, or manuscript sources.

## What is implemented

- Seven fixed candidate tokens per frame: current Full fusion, Nano, HistAB/MIS,
  previous frame, and Nano–HistAB 0.25/0.50/0.75 interpolation boxes.
- 22 inference-safe features per token, including normalized geometry, source
  confidence, contour/histogram/size/temporal scores, reliability, risk, motion,
  candidate agreement, and domain. Ground truth is stored separately.
- Three trainable gates:
  - `LinearFuser` (23 parameters);
  - `TinyMLPFuser` (1,889 parameters);
  - `MicroTransformerFuser` (9,313 parameters; one layer, two heads, width 32).
- A 45,830-parameter `VisualQualityResidualFuser` with:
  - synchronized template/current-ROI grayscale, edge, template-difference, and
    HSV back-projection maps;
  - candidate IoU-quality prediction;
  - shared visual backbone with separate chess/TrackVes heads;
  - bounded continuous center/size correction outside the candidate convex hull;
  - sequence-balanced sampling and target-domain validation calibration.
- The selected 46,046-parameter dynamic variant adds an inference-safe EMA
  template and two adaptive-template response channels.
- Advanced ablations additionally implement a frame-level GRU safety/size
  calibrator, 25-box local expansion, dense every-frame Nano consistency GRU,
  real-mask heads, frozen-response hooks, and independent-sequence validation.
- Frozen Full fusion, Nano, HistAB, score heuristic, conservative domain router,
  and GT-only oracle baselines.
- Sequence-level dual-domain OOF: 9 TrackVes folds and 11 chess folds. Every
  fold uses one target-domain test sequence and validation sequences from both
  domains. Normalization, early stopping, checkpoint choice, and residual
  calibration are fitted only on train/validation data.

## Reproduce

From this directory:

```bat
run_all.bat
```

Or run each stage:

```powershell
python candidate_exporter.py --config config/default.yaml
python run_nested_oof.py --config config/default.yaml
python evaluate_oof.py
python make_figures.py
python visual_feature_exporter.py --config config/default.yaml
python run_visual_oof.py --config config/default.yaml
python evaluate_visual_oof.py
python train_final_visual.py --config config/default.yaml
```

`candidate_exporter.py` consumes immutable per-frame TrackVes intermediate JSON
and aligned chess result CSVs from completed frozen tracker runs. The chess
sequence names and all source paths are recorded in the dataset manifest.

## Outputs

- `artifacts/candidate_dataset.npz`: unified 9,765-frame dataset.
- `artifacts/dataset_manifest.json`: schema and sequence inventory.
- `artifacts/checkpoints/<fold>/`: 60 fold-specific checkpoints.
- `artifacts/oof_predictions.csv`: predictions for all eight methods.
- `artifacts/run_manifest.json`: splits, seeds, commands, model histories,
  checkpoint hashes, versions, parameter counts, and timing.
- `artifacts/evaluation/`: aggregate/sequence metrics, corrected Wilcoxon tests,
  LaTeX table, and PNG/PDF figures.
- `RESULTS_REPORT.md`: interpretation and paper-use caveats.
- `artifacts/visual_responses.npz`: synchronized response maps for all frames.
- `artifacts/visual_checkpoints/<fold>/`: 20 visual fold checkpoints.
- `artifacts/visual_oof_predictions.csv`: complete visual OOF predictions.
- `artifacts/visual_evaluation/` and `VISUAL_RESULTS_REPORT.md`: visual model,
  continuous-oracle, significance, timing, figure, and LaTeX outputs.
- `artifacts/deployment/hybrid_visual_chess.pt`: all-sequence deployment
  checkpoint; load it with `HybridVisualRouterTracker`.
- `artifacts/deployment/hybrid_dynamic_chess.pt`: selected adaptive-template
  deployment checkpoint.
- `artifacts/deployment/chess_mask_head.pt`: selected chess real-mask head.
- `ADVANCED_RESULTS_REPORT.md`: all positive and negative advanced ablations.

## Result

Scalar-only learned models did not beat Full fusion. Static visual responses
raised equal-domain sequence-mean BBox-IoU to 0.6031. The selected dynamic
hybrid uses adaptive-template visual residual fusion on chess and Nano on
TrackVes, reaching 0.6083 versus 0.5961 for the previous domain router and
0.5956 for Full fusion. On chess it reaches 0.7302 versus 0.7058 for Full
fusion and 0.7296 for the discrete candidate oracle; 9/11 sequences improve.
The primary one-sided sequence Wilcoxon p-value is 0.0337, but it is exploratory
because the variant was selected after ablations. TrackVes retains Nano's best
tested result rather than applying a harmful visual correction.

The selected dynamic overhead is approximately 3.9 ms/frame on the current
machine (about 3.75 ms response preprocessing and 0.15 ms network inference), excluding
the unchanged frozen tracker runtime.

The chess mask head reaches ROI Mask-IoU 0.7740 versus 0.4824 for the bbox mask.
The TrackVes polygon-mask head is worse (0.1639 versus 0.2210), so deployment
routes chess to the learned mask and TrackVes to the bbox mask. Dense-frame
motion consistency improves Nano on all nine TrackVes sequences under its
sequential protocol (macro 0.3860 versus 0.3834).

OpenCV TrackerNano does not expose its true Siamese dense correlation map.
`internal_response_hook.py` captures the real Scheme-C Gaussian prior and any
returned hierarchy arrays when those paths execute; derived template responses
remain explicitly labelled as derived. The external-sequence adapter is ready,
but no unsupplied real sequence is fabricated.

Mask/Cov-IoU is intentionally not synthesized: combining boxes does not produce
a new segmentation mask. Existing paper files are untouched.

# Contour-aware ROI tracking of deforming soft tissue

Code and reproducibility files for the manuscript

**Contour-Aware Region-of-Interest Tracking of Deforming Soft Tissue in Endoscopic Surgical Navigation**

submitted to *Sensors* (MDPI).

This repository provides the online tracker, conservative and learned-gate extensions, configuration templates, analysis scripts, and aggregate evaluation summaries reported in the paper. It does not redistribute third-party images, clinical recordings, or the controlled-benchmark source videos.

## Scope

The method tracks a region of interest (ROI) around a deforming tissue area under the TrackVes safety-area protocol. It does **not** estimate dense point correspondences, tissue displacement fields, or surface strain.

One pipeline is used throughout:

| Name | What it is |
|---|---|
| **Ours** (`ours_scheme_c_nano`) | NanoTrack spatial prior + ContourSimilarityNet + HSV-histogram (HistAB) verification |
| **Ours-Conservative** | Fixed-rule size and re-detection arbitration around NanoTrack |
| **Ours-LearnedGate** | Supervised leave-one-sequence-out quality gate |
| **Ours-Contour** | Contour-based region delineation used in the segmentation comparison |

## Reported headline results

Numbers below are copied from the released evaluation logs. They are **not** a claim of uniform superiority over NanoTrack.

| Setting | Metric | Leading method | Score | Comparator | Note |
|---|---|---|---:|---|---|
| CholecSeg8k, zero-shot/prompted | Dice | Ours-Contour | 0.769 ± 0.13 | MedSAM 0.736 | paired Wilcoxon *p* < 0.01 |
| Controlled *N* = 11 | Std-IoU | NanoTrack | 0.712 ± 0.10 | Ours 0.706 | overlap |
| Controlled, official TrackEval | HOTA / AssA | Ours | 0.645 / 0.710 | NanoTrack 0.629 / 0.667 | association |
| TrackVes, 9 sequences | BBox-IoU | Ours-LearnedGate | 0.520 ± 0.22 | NanoTrack 0.492 | *p* = 0.50, not significant |

Full tables, sequence-level summaries, and split manifests are in [`reproducibility/`](reproducibility/).

## Public datasets (obtain from the original providers)

- [CholecSeg8k](https://arxiv.org/abs/2012.12453)
- [Kvasir-Instrument](https://datasets.simula.no/kvasir-instrument/)
- [TrackVes](https://doi.org/10.5281/zenodo.822053)
- [COCO 2017](https://cocodataset.org/) (contour pseudo-label pretraining only)

The controlled chessboard sequences are not redistributed. Ownership and redistribution permission have not been documented.

## Environment

Python 3.10 or later. Install the packages listed in `requirements.txt`:

```text
pip install -r requirements.txt
```

OpenCV NanoTrack ONNX weights and a ContourSimilarityNet checkpoint are required for the online path. Place them according to `config/default.example.yaml` (copy to `config/default.yaml` and edit local paths).

## Reproduce the online evaluations

From this directory:

```bat
run_all.bat
```

Or run stages separately:

```text
python run_trackves_online.py --config config/default.yaml
python run_trackves_conservative.py --config config/default.yaml
python run_trackves_global_gate_online.py --config config/default.yaml
python run_chess_online.py --config config/default.yaml
```

Sequence-level out-of-fold splits are recorded in

- `reproducibility/oof_split_manifest.json`
- `reproducibility/visual_oof_split_manifest.json`

Do not retune thresholds on a held-out test sequence if you intend to match the paper protocol.

## What is not claimed

- Uniform improvement over NanoTrack on every overlap metric
- Dense deformation-field or point-trajectory accuracy
- AR/MR navigation endpoint accuracy
- Strict zero-shot TrackVes evaluation for original Ours and Ours-Conservative (the frozen ContourSimilarityNet has in-domain TrackVes supervision; see the paper)

## Citation

If you use this code, please cite the Sensors manuscript once it is published. Until then, cite this repository and the public datasets above.

## License

The source files in this repository are released under the MIT License (see `LICENSE`). Third-party datasets remain under their original terms.

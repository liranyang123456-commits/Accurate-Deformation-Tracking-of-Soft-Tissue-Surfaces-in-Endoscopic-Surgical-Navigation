# Nano–HistAB–ContourSim learned fusion: OOF results

This report contains only sequence-level out-of-fold results. Ground truth was used for training loss and offline metrics, never as an inference feature.

## Model selection

- Selected deployable method: `domain_router`
- Best learned model: `linear`
- Learned model improves equal-domain BBox-IoU: `False`
- Significant in both domains after Holm correction: `False`
- Eligible to be named the learned-fusion winner: `False`

## OOF summary

| Domain | Method | BBox-IoU | Success@0.5 | Precision@20 | CE (px) | Params | Fuser ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|
| chess | oracle | 0.7296 | 0.8829 | 0.6609 | 44.95 | - | - |
| chess | domain_router | 0.7058 | 0.8753 | 0.6473 | 70.97 | - | - |
| chess | full_fusion | 0.7058 | 0.8753 | 0.6473 | 70.97 | - | - |
| chess | linear | 0.7042 | 0.8753 | 0.6425 | 71.12 | 23 | 0.0232 |
| chess | tiny_mlp | 0.7008 | 0.8690 | 0.6414 | 71.94 | 1889 | 0.0208 |
| chess | nano | 0.7005 | 0.8738 | 0.6482 | 97.80 | - | - |
| chess | micro_transformer | 0.6949 | 0.8656 | 0.6374 | 72.38 | 9313 | 0.0283 |
| chess | heuristic_score | 0.3572 | 0.3341 | 0.1574 | 235.80 | - | - |
| chess | hist | 0.2490 | 0.1921 | 0.0428 | 314.99 | - | - |
| trackves | oracle | 0.5070 | 0.5721 | 0.3693 | 108.79 | - | - |
| trackves | domain_router | 0.4864 | 0.5568 | 0.3555 | 114.47 | - | - |
| trackves | nano | 0.4864 | 0.5568 | 0.3555 | 114.47 | - | - |
| trackves | full_fusion | 0.4854 | 0.5563 | 0.3566 | 111.95 | - | - |
| trackves | linear | 0.4852 | 0.5563 | 0.3487 | 111.80 | 23 | 0.0232 |
| trackves | tiny_mlp | 0.4843 | 0.5475 | 0.3472 | 109.68 | 1889 | 0.0208 |
| trackves | micro_transformer | 0.4842 | 0.5489 | 0.3463 | 110.55 | 9313 | 0.0283 |
| trackves | heuristic_score | 0.4800 | 0.5376 | 0.3489 | 112.53 | - | - |
| trackves | hist | 0.4763 | 0.5267 | 0.3474 | 107.66 | - | - |

## Statistical and scope notes

- Wilcoxon tests use paired sequence means, not correlated frame samples.
- `oracle` is a GT-only upper bound and is never deployable.
- Reported latency is the fusion gate alone; candidate tracker runtime is unchanged.
- A new Mask/Cov-IoU is not reported: convex box fusion does not produce a new segmentation mask. Reusing a source mask would not be a valid learned-box metric.
- Full split records, checkpoint hashes, seeds, and commands are in `artifacts/run_manifest.json`.

See `artifacts/evaluation/wilcoxon_vs_full_fusion.csv` for all corrected tests.

# Final online and OOF evaluation report

## Cross-dataset summary and comparison with manuscript Ours

| Dataset / protocol | Method variant | BBox-IoU | Mask-IoU | Success@0.5 | Precision@20px | Speed |
|---|---|---:|---:|---:|---:|---:|
| Chessboard, manuscript, 11-sequence mean | Ours (scheme-C/nano) | 0.706 | not reported | 0.876 | 0.648 | not reported |
| Chessboard, strict OOF, 11-sequence mean | Ours + dynamic visual residual | 0.7302 | 0.7740 (ROI mask-head evaluation) | not recomputed | not recomputed | not benchmarked |
| Chessboard, final online, 22-board-unit mean | Ours + dynamic visual residual + mask router | 0.7345 | 0.7846 | 0.8978 | 0.6158 | 0.39 FPS |
| TrackVes, manuscript, 9-sequence mean | Ours (scheme-C/nano) | 0.489 | 0.329 | 0.558 | not reported | not reported |
| TrackVes, manuscript best baseline | NanoTrack | 0.492 | 0.331 | 0.560 | not reported | not reported |
| TrackVes, final online, fixed arbitration/no box-gate fitting | Ours-Conservative | **0.4947** | 0.3304 | 0.5648 | 0.3524 | 5.12 FPS |
| TrackVes, strict OOF, 9-sequence mean | Ours-LearnedGate | 0.5152 | not evaluated | not recomputed | not recomputed | not benchmarked |
| TrackVes, final cross-fitted online, 9-sequence mean | Ours-LearnedGate | **0.5201** | **0.3390** | **0.6044** | **0.3530** | **6.85 FPS** |

The Chessboard online row averages 22 board units, whereas the manuscript row
first averages boards within each of 11 sequences. The strict OOF value
(0.7302) is the directly comparable sequence-level estimate. TrackVes rows use
the same 9-sequence mean protocol.

## Chessboard online benchmark

- Protocol: first-frame GT initialization, then online state updates.
- Coverage: 11 sequences, 22 board units, 3,848 evaluated frames.
- BBox-IoU: **0.7345**.
- Mask-IoU: **0.7846**.
- Success@0.5: **0.8978**.
- Precision@20px: **0.6158**.
- End-to-end runtime: 9,788.7 s (**0.39 FPS**).
- The prior Full-fusion OOF reference was 0.7058 BBox-IoU; the online
  dynamic model is +0.0287 higher, but this comparison mixes online final
  training and OOF estimates and is therefore descriptive rather than a
  statistical test.

## TrackVes original-protocol online benchmark

- Coverage: 9 sequences, 5,931 GT frames.
- Selected deployed route: NanoTrack (the visual residual branch was rejected
  on TrackVes validation).
- BBox-IoU: **0.4910**.
- Mask-IoU: **0.3294**.
- Success@0.5: **0.5627**.
- Precision@20px: **0.3469**.
- End-to-end runtime: 771.4 s (**7.69 FPS**).
- Paper Nano baseline: 0.4922 BBox-IoU. The online router therefore reproduces
  Nano within run/decoder variation and does not surpass it.

## Non-learned conservative Scheme-C extension

- NanoTrack is the default output. HistAB, ContourSimilarityNet, and size
  consistency jointly verify center-preserving scale candidates limited to
  +/-10%; no TrackVes tracking outcome is used to fit the arbitration
  thresholds.
- The inherited ContourSimilarityNet checkpoint is not label-free: its released
  fine-tuning entry point partitions all nine TrackVes sequences into
  sequence-disjoint train/validation subsets. Original Ours and
  Ours-Conservative are therefore in-domain evaluations, not strict zero-shot
  or test-sequence-excluded estimates.
- A static first-frame global template is evaluated only periodically or after
  persistent risk, and requires NCC peak, peak-to-sidelobe ratio, peak margin,
  HistAB, ContourSimilarityNet, and size checks before replacing Nano.
- Online 9-sequence BBox-IoU: **0.4947**, versus Nano **0.4922**
  (+0.00254, +0.52% relative).
- Five sequences improve, two degrade, and two tie; one-sided sequence
  Wilcoxon **p=0.1484**.
- Mask-IoU: **0.3304**; Success@0.5: **0.5648**; Precision@20px:
  **0.3524**; runtime: **5.12 FPS**.
- This is the first non-learned Scheme-C variant in these experiments to
  exceed Nano's mean, although the margin is small and not significant.

## New global re-detection optimization

- Static global template candidates materially improve EV3 (0.2427 to 0.5123)
  and IV1 (0.1280 to 0.2414), but harm several other sequences.
- Scalar LOSO safety gate: 0.4828; rejected.
- Stacked temporal gate: 0.4866; rejected.
- Visual quality gate: 17,073 parameters, strict sequence LOSO.
- Visual-gate BBox-IoU: **0.5152**, versus same-fold Nano **0.4864**
  (+0.0289).
- Sequence-level one-sided Wilcoxon: **p=0.3438** (3 improved, 4 degraded,
  2 ties).
- The cross-fitted gates were then run as actual online trackers under the
  original 9-sequence protocol. Online BBox-IoU is **0.5201**, Mask-IoU is
  **0.3390**, Success@0.5 is **0.6044**, and Precision@20px is **0.3530**.
- Against the paper Nano baseline (0.4922), the online gain is **+0.0279**
  (**+5.67% relative**). Five sequences improve and four degrade; the
  sequence-level one-sided Wilcoxon test is **p=0.5000**.
- Online global-gate runtime is 865.3 s for 5,931 evaluated frames
  (**6.85 FPS**).

## Defensible conclusion

The final method is strongest on the Chessboard benchmark. On TrackVes, the
test-sequence-excluded online Ours-LearnedGate reaches the highest mean BBox-IoU among
the methods tested here (0.5201 versus Nano 0.4922). It is therefore the
benchmark leader by mean score, but the gain is driven mainly by EV3 and is not
significant across nine sequences. The defensible wording is "best mean result
among tested methods", not a statistically established universal SOTA claim.

## Three-pass manuscript audit

1. Figure provenance: the legacy clinical collage `FigTrackingValidation.jpg`
   had no traceable current prediction CSV or generation script and was removed
   from the manuscript. Tracking figures now use exported GT/prediction logs.
2. Numerical audit: all TrackVes means and Wilcoxon tests were independently
   recomputed from the nine sequence summaries. Extension-table dispersions were
   corrected to sample SD (`ddof=1`) to match the original TrackVes table.
3. Logic/protocol audit: Ours, Ours-Conservative, and Ours-LearnedGate are
   separated as frozen, fixed-rule, and supervised LOSO variants. The manuscript
   reports the learned result as the best tested mean, not universal SOTA.
4. Supervision audit: the TrackVes training/validation overlap of the frozen
   ContourSimilarityNet verifier is now disclosed in Methods, dataset protocol,
   the TrackVes table caption, Discussion, Conclusions, and this report. Claims
   of strict external or zero-shot validation were removed for original Ours
   and Ours-Conservative.

# Visual quality + continuous residual fusion

- Selected deployable: `hybrid_dynamic_router`
- Quality Spearman correlation: 0.6405
- Candidate top-1 accuracy: 0.4498
- Visual preprocessing: 3.755 ms/frame
- Visual model: 0.151 ms/frame
- Visual model beats domain router: `True`
- Selected improvement significant in both domains: `False`
- Chess primary one-sided Wilcoxon p: 0.0337 (exploratory after model selection)
- Chess visual BBox-IoU: 0.7302 (Full fusion 0.7058)
- TrackVes hybrid BBox-IoU: 0.4864

## Equal-domain sequence-mean BBox-IoU

- `continuous_oracle`: 0.8204
- `oracle`: 0.6183
- `hybrid_dynamic_router`: 0.6083
- `visual_dynamic`: 0.6071
- `domain_router`: 0.5961
- `full_fusion`: 0.5956

The continuous oracle is GT-only and measures headroom after permitting bounded box correction outside the original convex candidate set.
All reported visual-model predictions are sequence-level OOF.
Six static+adaptive-template response channels are synchronously derived from the image and initial template; they are not claimed to be internal NanoTrack response maps.

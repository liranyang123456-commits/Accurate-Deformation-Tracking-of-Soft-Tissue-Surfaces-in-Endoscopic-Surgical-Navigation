# Advanced fusion implementation and test report

- Selected protocol-SOTA deployment: `hybrid_dynamic_router`
- Equal-domain BBox-IoU: 0.6083
- Chess dynamic BBox-IoU: 0.7302
- Chess Full fusion: 0.7058
- Improved chess sequences: 9/11
- Exploratory one-sided Wilcoxon p: 0.0337
- TrackVes routed Nano BBox-IoU: 0.4864

## Tested additions

- Frame safety + GRU + size calibration: 0.5953; rejected.
- TrackVes size-aware dynamic visual router: 0.6080; rejected.
- Mask domain router: 0.4975 versus equal-domain bbox-mask 0.3517; selected for mask output.
- Chess real-mask head: 0.7740; TrackVes polygon head 0.1639 was rejected in favor of bbox mask 0.2210.
- Expanded candidates: TrackVes oracle 0.3494, chess oracle 0.7726.
- Internal-response hook implemented; OpenCV Nano dense score map is unavailable.
- Dense every-frame consistency GRU: 0.2505 versus Nano 0.2484; improved all 9 sequences.
- Independent-sequence adapter validated; no new real sequence was supplied, so none was fabricated or included.

Only the best OOF variant is selected. Negative ablations are retained for reproducibility.

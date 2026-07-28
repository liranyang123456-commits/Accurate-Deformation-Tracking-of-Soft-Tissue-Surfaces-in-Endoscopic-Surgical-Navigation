# Reproducibility materials

This directory contains non-image, non-identifiable artifacts supporting the reported Ours methods:

- `dataset_manifest.json`: sequence inventory and inference-feature schema.
- `oof_split_manifest.json` and `visual_oof_split_manifest.json`: sequence-level OOF partitions.
- `online_trackves_conservative_summary.json` and `online_trackves_global_gate_summary.json`: aggregate and sequence-level online TrackVes metrics.
- `online_chess_summary.json`: aggregate and sequence-level controlled-benchmark metrics with local paths removed.

Raw third-party datasets, source images, annotations, and per-frame outputs are not redistributed here. Obtain CholecSeg8k, Kvasir-Instrument, TrackVes, and COCO from their official providers. The controlled benchmark remains unavailable pending documented ownership and redistribution permission.

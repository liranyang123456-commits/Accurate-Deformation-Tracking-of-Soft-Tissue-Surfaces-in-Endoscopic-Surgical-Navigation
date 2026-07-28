"""Train the final all-chess real-mask head using OOF-selected threshold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset_builder import load_arrays, sequence_balanced_weights
from run_mask_oof import LightMaskHead


def main():
    arrays = load_arrays(Path("artifacts/candidate_dataset.npz"))
    visual = np.load("artifacts/visual_responses_dynamic.npz", allow_pickle=False)
    mask_data = np.load("artifacts/trackves_roi_masks.npz", allow_pickle=False)
    indices = np.flatnonzero(
        (arrays["domain"] == "chess") & mask_data["valid"].astype(bool)
    )
    maps = visual["maps"][indices, 0]
    targets = mask_data["masks"][indices]
    dataset = TensorDataset(
        torch.from_numpy(maps).float().div(255),
        torch.from_numpy(targets).float(),
    )
    weights = sequence_balanced_weights(arrays, indices)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(indices), replacement=True,
        generator=torch.Generator().manual_seed(20260717),
    )
    loader = DataLoader(dataset, batch_size=256, sampler=sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightMaskHead(maps.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(18):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            y = y.to(device)
            probability = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            dice = 1 - (
                (2 * (probability * y).sum((1, 2)) + 1)
                / (probability.sum((1, 2)) + y.sum((1, 2)) + 1)
            ).mean()
            (bce + dice).backward()
            optimizer.step()
    oof = json.loads(
        Path("artifacts/mask_oof_predictions.manifest.json").read_text(encoding="utf-8")
    )
    thresholds = [
        x["threshold"] for x in oof["folds"] if x["fold_id"].startswith("chess_")
    ]
    threshold = float(np.median(thresholds))
    output = Path("artifacts/deployment/chess_mask_head.pt")
    torch.save({
        "state_dict": model.state_dict(), "threshold": threshold,
        "channels": int(maps.shape[1]), "roi_context": 1.5, "size": 24,
    }, output)
    print(output)


if __name__ == "__main__":
    main()

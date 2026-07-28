"""Lightweight candidate-box gating models."""
from __future__ import annotations

import torch
from torch import nn


class BaseFuser(nn.Module):
    def weights(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, features: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(features)
        fused = (weights.unsqueeze(-1) * boxes).sum(dim=1)
        return fused, weights


class LinearFuser(BaseFuser):
    """Logistic candidate ranker; simplest learned baseline."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def weights(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(features).squeeze(-1), dim=1)


class TinyMLPFuser(BaseFuser):
    """Shared candidate MLP plus softmax gate (~2k parameters)."""
    def __init__(self, feature_dim: int, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def weights(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(features).squeeze(-1), dim=1)


class MicroTransformerFuser(BaseFuser):
    """One-layer, two-head candidate-token transformer."""
    def __init__(
        self, feature_dim: int, d_model: int = 32, heads: int = 2, layers: int = 1
    ):
        super().__init__()
        self.embed = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=64,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.score = nn.Linear(d_model, 1)

    def weights(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.embed(features))
        return torch.softmax(self.score(tokens).squeeze(-1), dim=1)


def make_model(name: str, feature_dim: int, cfg: dict) -> BaseFuser:
    if name == "linear":
        return LinearFuser(feature_dim)
    if name == "tiny_mlp":
        return TinyMLPFuser(feature_dim, int(cfg["models"]["mlp_hidden"]))
    if name == "micro_transformer":
        return MicroTransformerFuser(
            feature_dim, int(cfg["models"]["transformer_dim"]),
            int(cfg["models"]["transformer_heads"]),
            int(cfg["models"]["transformer_layers"]),
        )
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

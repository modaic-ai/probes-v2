from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 1


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings with shape [batch, hidden], got {tuple(embeddings.shape)}"
            )
        embeddings = embeddings.to(dtype=self.linear.weight.dtype)
        return self.linear(self.dropout(embeddings)).squeeze(-1)


def _checkpoint_payload(
    head: LinearProbeHead,
    base_model: str,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "linear_probe_head",
        "base_model": base_model,
        "input_dim": int(head.linear.in_features),
        "dropout": float(head.dropout.p),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in head.linear.state_dict().items()
        },
    }


def save_linear_head_checkpoint(
    path: str | Path,
    head: LinearProbeHead,
    base_model: str,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(head, base_model), checkpoint_path)


def load_linear_head_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a valid mapping")

    required_keys = {
        "format_version",
        "kind",
        "base_model",
        "input_dim",
        "state_dict",
    }
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing required keys: {', '.join(missing)}"
        )
    if payload["kind"] != "linear_probe_head":
        raise ValueError(
            f"Unsupported checkpoint kind '{payload['kind']}' in {checkpoint_path}"
        )
    if int(payload["input_dim"]) < 1:
        raise ValueError(
            f"Checkpoint input_dim must be >= 1, got {payload['input_dim']}"
        )
    return payload


def build_linear_head_from_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[LinearProbeHead, dict[str, Any]]:
    payload = load_linear_head_checkpoint(path)
    head = LinearProbeHead(
        input_dim=int(payload["input_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    head.linear.load_state_dict(payload["state_dict"])
    if device is not None:
        head.to(device)
    head.eval()
    return head, payload

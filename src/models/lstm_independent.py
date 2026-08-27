"""Per-asset LSTM: no graph. Baseline that ignores connectedness."""
from __future__ import annotations
import torch
from torch import nn

class IndependentLSTM(nn.Module):
    def __init__(self, hidden: int = 16, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) returns. Output logits (B, N)."""
        b, t, n = x.shape
        z = x.transpose(1, 2).reshape(b * n, t, 1)
        out, _ = self.lstm(z)
        logits = self.head(out[:, -1, :])
        return logits.reshape(b, n)

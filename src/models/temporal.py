"""GRU over node embeddings from a dense GCN at each window (EvolveGCN-like).

Not EvolveGCN-H/O from Pareja et al. (2019) in full; a small GRU mixes
per-node GCN states across time so that the model can use a sequence of
graphs rather than one static snapshot. MTGNN/GAT are cited as related
families, not implemented here.
"""
from __future__ import annotations
import torch
from torch import nn
from .static_gcn import normalised_adj

class TemporalGRUGcn(nn.Module):
    def __init__(self, in_dim: int = 4, hid: int = 16) -> None:
        super().__init__()
        self.gcn = nn.Linear(in_dim, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)
        self.act = nn.ReLU()

    def forward(self, x_seq: torch.Tensor, adj_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (W, N, F) windows. adj_seq: (W, N, N). logits (N,)."""
        w, n, f = x_seq.shape
        hs = []
        for k in range(w):
            a = normalised_adj(adj_seq[k])
            h = self.act(a @ self.gcn(x_seq[k]))  # (N, hid)
            hs.append(h)
        seq = torch.stack(hs, dim=1)  # (N, W, hid)
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :]).squeeze(-1)

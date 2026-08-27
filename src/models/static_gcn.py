"""Dense-adjacency GCN (Kipf-Welling in spirit). No torch_geometric."""
from __future__ import annotations
import torch
from torch import nn

def normalised_adj(a: torch.Tensor) -> torch.Tensor:
    """A_hat = D^{-1/2} (A+I) D^{-1/2} if A has no self loops already added."""
    n = a.size(-1)
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    a = a + eye
    deg = a.sum(dim=-1).clamp(min=1e-6)
    dinv = torch.diag(deg.pow(-0.5))
    return dinv @ a @ dinv

class DenseGCN(nn.Module):
    def __init__(self, in_dim: int = 8, hid: int = 16) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid)
        self.w2 = nn.Linear(hid, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (N, F) node features at one snapshot. adj: (N, N). logits (N,)."""
        a = normalised_adj(adj)
        h = self.act(a @ self.w1(x))
        return self.w2(a @ h).squeeze(-1)

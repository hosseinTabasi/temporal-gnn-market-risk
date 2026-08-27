"""Graph construction for a small equity universe.

All edge features at time t use returns with index <= t. Lead-lag is
estimated on a trailing window that stops at t. This is connectedness
in the sense of Billio et al. / Diebold-Yilmaz, not a causal contagion
graph.
"""
from __future__ import annotations
import numpy as np

TICKERS_TOY = [f"T{i:02d}" for i in range(15)]
# Three dummy "sectors" of five names each.
SECTOR = {t: i // 5 for i, t in enumerate(TICKERS_TOY)}

def _window(returns: np.ndarray, t: int, lookback: int) -> np.ndarray:
    """returns: (T, N). Slice (max(0,t-lookback+1) .. t) inclusive."""
    start = max(0, t - lookback + 1)
    return returns[start : t + 1]

def correlation_graph(returns: np.ndarray, t: int, lookback: int = 20,
                      thresh: float = 0.3) -> np.ndarray:
    """Rolling Pearson correlation adjacency at t (data <= t)."""
    w = _window(returns, t, lookback)
    n = returns.shape[1]
    if w.shape[0] < 3:
        return np.eye(n, dtype=np.float32)
    c = np.corrcoef(w.T)
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 0.0)
    a = (np.abs(c) >= thresh).astype(np.float32)
    a = np.maximum(a, np.eye(n, dtype=np.float32))
    return a

def leadlag_graph(returns: np.ndarray, t: int, lookback: int = 20,
                  lag: int = 1, thresh: float = 0.15) -> np.ndarray:
    """Directed lead-lag: corr(r_i[s], r_j[s+lag]) on s with s+lag <= t."""
    n = returns.shape[1]
    start = max(0, t - lookback + 1)
    # Use pairs (s, s+lag) both <= t.
    if t - lag < start:
        return np.eye(n, dtype=np.float32)
    a = np.eye(n, dtype=np.float32)
    left = returns[start : t - lag + 1]   # <= t-lag
    right = returns[start + lag : t + 1]  # <= t
    if left.shape[0] < 3:
        return a
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = np.corrcoef(left[:, i], right[:, j])[0, 1]
            if np.isfinite(c) and abs(c) >= thresh:
                a[i, j] = 1.0
    return a

def sector_graph(n: int = 15) -> np.ndarray:
    """Undirected dummy GICS-like blocks (5+5+5)."""
    a = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i // 5 == j // 5:
                a[i, j] = 1.0
    return a

def graphs_upto_t(returns: np.ndarray, t: int, lookback: int = 20) -> dict:
    """Bundle used by models. Every matrix is a function of data <= t."""
    return {
        "corr": correlation_graph(returns, t, lookback=lookback),
        "leadlag": leadlag_graph(returns, t, lookback=lookback),
        "sector": sector_graph(returns.shape[1]),
        "t": t,
        "max_index_used": t,
    }

def assert_no_future(returns: np.ndarray, t: int, adj: np.ndarray, lookback: int = 20) -> None:
    """Recompute with returns after t zeroed; adjacency must match."""
    frozen = returns.copy()
    frozen[t + 1 :] = 0.0
    a2 = correlation_graph(frozen, t, lookback=lookback)
    if not np.allclose(adj, a2):
        raise AssertionError("edge features at t used returns after t")

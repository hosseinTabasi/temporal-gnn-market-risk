"""Edge features at t must not use returns after t. AUPRC is defined."""
from __future__ import annotations
import numpy as np
import torch
from graphs.build import correlation_graph, leadlag_graph, graphs_upto_t, assert_no_future
from metrics.auprc import auprc, crash_labels
from models.lstm_independent import IndependentLSTM
from models.static_gcn import DenseGCN
from models.temporal import TemporalGRUGcn
from data_panel import synthetic_panel

def test_corr_no_future():
    pan = synthetic_panel(n_steps=120, seed=0)
    r = pan["returns"]
    t = 50
    adj = correlation_graph(r, t, lookback=20)
    assert_no_future(r, t, adj, lookback=20)
    # Mutating the future must not change adj
    r2 = r.copy(); r2[t+1:] = 99.0
    adj2 = correlation_graph(r2, t, lookback=20)
    assert np.allclose(adj, adj2)

def test_leadlag_uses_only_past():
    pan = synthetic_panel(n_steps=120, seed=1)
    r = pan["returns"]
    t = 60
    a = leadlag_graph(r, t, lookback=20, lag=1)
    r2 = r.copy(); r2[t+1:] = -99.0
    a2 = leadlag_graph(r2, t, lookback=20, lag=1)
    assert np.allclose(a, a2)
    g = graphs_upto_t(r, t)
    assert g["max_index_used"] == t

def test_auprc_defined():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.05, 0.4, 0.2])
    v = auprc(y, s)
    assert np.isfinite(v) and 0.0 <= v <= 1.0
    pan = synthetic_panel(seed=0)
    lab = crash_labels(pan["returns"], q=0.05)
    assert lab.mean() < 0.15  # rare-ish

def test_model_shapes():
    pan = synthetic_panel(n_steps=40, seed=0)
    r = pan["returns"]
    x = torch.from_numpy(r[None, ...])  # (1, T, N)
    lstm = IndependentLSTM()
    out = lstm(x)
    assert out.shape == (1, r.shape[1])
    n = r.shape[1]
    gcn = DenseGCN(in_dim=4)
    feat = torch.randn(n, 4)
    adj = torch.from_numpy(correlation_graph(r, t=20))
    logits = gcn(feat, adj)
    assert logits.shape == (n,)
    tg = TemporalGRUGcn(in_dim=4, hid=8)
    xs = torch.randn(5, n, 4)
    adjs = torch.stack([adj] * 5)
    y = tg(xs, adjs)
    assert y.shape == (n,)

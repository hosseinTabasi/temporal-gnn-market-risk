"""TOY: 1-epoch comparison LSTM vs GCN vs temporal GRU-GCN. Writes AUPRC CSV."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data_panel import synthetic_panel, try_nasdaq100_prices
from graphs.build import correlation_graph
from metrics.auprc import auprc, crash_labels
from models.lstm_independent import IndependentLSTM
from models.static_gcn import DenseGCN
from models.temporal import TemporalGRUGcn
from viz.hero import write_hero

def _node_feat(returns, t, w=8):
    sl = returns[max(0, t-w+1): t+1]
    mu = sl.mean(axis=0); sd = sl.std(axis=0)
    last = returns[t]
    mom = sl[-1] - sl[0] if sl.shape[0] > 1 else last
    return np.stack([mu, sd, last, mom], axis=1).astype(np.float32)

def main(root: Path | None = None) -> dict:
    root = Path(root) if root else Path.cwd()
    torch.manual_seed(0); np.random.seed(0)
    pan = synthetic_panel(n_steps=120, seed=0, shock_t=80)
    r = pan["returns"]
    y = crash_labels(r, q=0.05)
    nd, prices = try_nasdaq100_prices(root)
    src_note = pan["source"] if nd is None else prices
    # Train on t=20..79, eval t=80..100 (includes the TOY shock).
    lstm = IndependentLSTM(hidden=8)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-2)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(1):
        for t in range(20, 80):
            xb = torch.from_numpy(r[t-16:t][None, ...])  # (1,16,N)
            yb = torch.from_numpy(y[t].astype(np.float32))[None, :]
            opt.zero_grad()
            loss = crit(lstm(xb), yb)
            loss.backward(); opt.step()
    scores_lstm = []
    ys = []
    with torch.no_grad():
        for t in range(80, 101):
            xb = torch.from_numpy(r[t-16:t][None, ...])
            scores_lstm.append(torch.sigmoid(lstm(xb)).numpy().ravel())
            ys.append(y[t])
    a_lstm = auprc(np.stack(ys), np.stack(scores_lstm))
    # Static GCN at last train snapshot, one epoch over t=40..79
    gcn = DenseGCN(in_dim=4, hid=8)
    optg = torch.optim.Adam(gcn.parameters(), lr=1e-2)
    for _ in range(1):
        for t in range(40, 80):
            feat = torch.from_numpy(_node_feat(r, t-1))
            adj = torch.from_numpy(correlation_graph(r, t-1))
            yb = torch.from_numpy(y[t].astype(np.float32))
            optg.zero_grad()
            loss = crit(gcn(feat, adj), yb)
            loss.backward(); optg.step()
    scores_g = []
    with torch.no_grad():
        for t in range(80, 101):
            feat = torch.from_numpy(_node_feat(r, t-1))
            adj = torch.from_numpy(correlation_graph(r, t-1))
            scores_g.append(torch.sigmoid(gcn(feat, adj)).numpy())
    a_gcn = auprc(np.stack(ys), np.stack(scores_g))
    # Temporal: 5-window GRU-GCN, one epoch
    tg = TemporalGRUGcn(in_dim=4, hid=8)
    optt = torch.optim.Adam(tg.parameters(), lr=1e-2)
    for _ in range(1):
        for t in range(40, 80):
            xs = torch.stack([torch.from_numpy(_node_feat(r, t-5+k)) for k in range(5)])
            adjs = torch.stack([torch.from_numpy(correlation_graph(r, t-5+k)) for k in range(5)])
            yb = torch.from_numpy(y[t].astype(np.float32))
            optt.zero_grad()
            loss = crit(tg(xs, adjs), yb)
            loss.backward(); optt.step()
    scores_t = []
    with torch.no_grad():
        for t in range(80, 101):
            xs = torch.stack([torch.from_numpy(_node_feat(r, t-5+k)) for k in range(5)])
            adjs = torch.stack([torch.from_numpy(correlation_graph(r, t-5+k)) for k in range(5)])
            scores_t.append(torch.sigmoid(tg(xs, adjs)).numpy())
    a_tg = auprc(np.stack(ys), np.stack(scores_t))
    adj80 = correlation_graph(r, 80)
    fig = write_hero(r, adj80, 80, root / "results" / "figures" / "toy_shock.png")
    row = {
        "label": "TOY", "source": src_note, "shock_t": 80,
        "auprc_lstm": None if np.isnan(a_lstm) else round(float(a_lstm), 6),
        "auprc_gcn": None if np.isnan(a_gcn) else round(float(a_gcn), 6),
        "auprc_temporal": None if np.isnan(a_tg) else round(float(a_tg), 6),
        "hero": str(fig),
        "note": "TOY synthetic shock at t=80; not COVID/SVB; connectedness not contagion",
    }
    out = root / "results" / "tables" / "toy_auprc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerow(row)
    return row

if __name__ == "__main__":
    print(main())

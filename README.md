# Temporal GNNs for Cross-Asset Risk and Connectedness

**Author:** Hossein Tabasi (2026). MIT licence. Junior research scaffold.

## Question

Can a temporal GNN on rolling-correlation, lead-lag, and sector graphs
forecast asset-level crash labels and market-stress episodes better than
independent LSTMs and a static GCN, as measured by AUPRC (rare events)?
We speak of **connectedness**, not contagion, and we do not claim that
an edge is a causal transmission channel.

## Why it matters

Billio, Getmansky, Lo and Pelizzon (2012) and Diebold and Yilmaz (2014)
measure cross-asset connectedness with Granger and forecast-error
variance decompositions. Graph neural nets (Kipf and Welling GCN, Velickovic
GAT, Wu et al. MTGNN, Pareja et al. EvolveGCN) offer a way to put those
graphs into a trainable filter. The empirical question is whether the
graph adds anything over a per-name LSTM on a rare crash label. AUPRC
is the metric because accuracy is dominated by the negative class.

## Data

Default universe: 15 synthetic tickers, 120 steps, a planted shock at
t=80 labelled **TOY**. Optional: a Nasdaq-100 public price panel
(`scripts/nasdaq100.md`); the download fails gracefully. No claim that
the toy shock is COVID-19 or SVB.

## Method

- Graphs at t use returns with index <= t only (rolling corr, lagged
  cross-corr, dummy sector blocks).
- Independent LSTM on each name's return window.
- Static dense GCN (Kipf-Welling normalisation; no torch_geometric
  required).
- Temporal GRU on a sequence of GCN node states (EvolveGCN-like).
- Labels: left-tail crash indicators. Metric: AUPRC.

## Baselines

Independent LSTM; static GCN; sector-only adjacency as a robustness
hook (the sector graph is always available).

## Results

**NO FULL RESULTS YET.** There is no Nasdaq-100 panel in this clone.
`results/tables/toy_auprc.csv` and `results/figures/toy_shock.png` are
**TOY** (synthetic shock at t=80). They are not COVID/SVB findings and
are not evidence of contagion.

## Limitations

Fifteen names. Dense GCN is not GAT. Lead-lag is correlation at lag 1,
not a Diebold-Yilmaz FEVD. Edges are not identified causal paths.
AUPRC on a planted shock overstates a real-crisis exercise.

## Reproduce (toy)

PYTHONPATH=src python -m run_toy
PYTHONPATH=src python -m pytest -q

## References

Kipf and Welling (ICLR 2017). Velickovic et al. GAT (ICLR 2018).
Wu et al. MTGNN (KDD 2020). Pareja et al. EvolveGCN (AAAI 2020).
Billio et al. (JFE 2012). Diebold and Yilmaz (J. Econometrics 2014).

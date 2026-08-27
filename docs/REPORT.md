# Workshop outline: Temporal GNNs for cross-asset connectedness and crash AUPRC

Hossein Tabasi, 2026. Methods outline. **No public price panel has been
successfully downloaded in this clone. All figures are TOY unless a
later run says otherwise.** Language: connectedness, not contagion.

## 1. Question

Equity returns co-move. That fact does not identify a transmission
network. Billio, Getmansky, Lo and Pelizzon (Journal of Financial
Economics, 2012) and Diebold and Yilmaz (Journal of Econometrics, 2014)
construct statistical connectedness measures — Granger causality
networks and forecast-error variance decompositions — that are
explicitly reduced-form. Graph neural networks (GNNs) take those
objects as input and ask a forecasting question: given a sequence of
graphs built from data up to t, can we rank names by the probability of
a left-tail return at t+h better than a model that treats names as
independent time series?

The comparison set is (i) an independent LSTM per name, (ii) a static
GCN on one snapshot, and (iii) a temporal model that GRUs over a
sequence of GCN states (in the spirit of EvolveGCN, not a line-by-line
reimplementation). The metric is AUPRC because crash labels are rare
and accuracy is almost a constant. We do not claim that an edge is a
causal contagion channel, and we do not claim a trading PnL.

## 2. Why "connectedness" not "contagion"

Contagion, in the Forbes and Rigobon sense, is an increase in
correlation after controlling for fundamentals, or a structural
spillover. A rolling Pearson matrix does not identify that. A GCN
layer is a linear smoother on a graph; it cannot, by itself, distinguish
common shocks from pairwise transmission. Using "contagion" in the title
would overclaim. Connectedness is the Diebold-Yilmaz word and the one
this repository uses in comments, figures, and the hero title.

## 3. Related architectures we cite but do not dump

Kipf and Welling (ICLR 2017) GCN is the static baseline; we implement
the D^{-1/2}(A+I)D^{-1/2} form on a dense adjacency so that
torch_geometric is optional. GAT (Velickovic et al., ICLR 2018) is cited
as the attention-on-edges alternative; it is not implemented in v0.1.
MTGNN (Wu et al., KDD 2020) mixes graph learning with temporal
convolution for multivariate series; we cite it as the closest
time-series GNN family. EvolveGCN (Pareja et al., AAAI 2019/2020)
evolves GCN weights with an RNN; our TemporalGRUGcn instead GRUs the
node states, which is a smaller cousin. None of these citations is a
licence to copy their codebases.

## 4. Data

The scientific target is a Nasdaq-100 daily return panel, with two
illustrative windows (2020-03 COVID, 2023-03 SVB) for subgraph figures.
Status: `try_nasdaq100_prices` fails closed. The runnable path is a
15-name synthetic factor panel of 120 steps with a planted common jump
at t=80, sector 0 hit harder. Tickers are T00..T14. That shock is
labelled TOY in every artefact. It is not a reconstruction of SVB.

## 5. Graphs (no future edges)

At each t we build three adjacencies using only returns[0:t] inclusive.

- Rolling correlation: Pearson on a lookback window, thresholded.
- Lead-lag: corr(r_i[s], r_j[s+lag]) for s+lag <= t. Directed.
- Sector: three dummy blocks of five names, a stand-in for GICS.

A unit test recomputes the correlation graph after zeroing or
exploding returns after t and requires equality. That test is the
methodological core of the repo: a GNN that peeks at t+1 is not a
forecast. Lead-lag is the easy place to leak; the slice is written so
that the right leg stops at t.

## 6. Labels and metric

A crash label is an asset-day in the left q-quantile of pooled returns
(default q=0.05). That is a rare-event definition, not a regulator's
definition of a circuit breaker. AUPRC is sklearn's average precision.
If a split has no positives, we return NaN rather than a fake 1.0.
Accuracy will not be the headline in any later table.

## 7. Models

IndependentLSTM maps each name's return window to a logit, sharing
weights across names but not mixing them. DenseGCN takes four node
features (window mean, std, last return, momentum) and one adjacency.
TemporalGRUGcn stacks five snapshots of (features, adj), applies a GCN
linear map at each, and GRUs along the window. Training in the toy
entry is one epoch on t=20..79, evaluation on t=80..100 which includes
the planted shock. That evaluation design is favourable to any model
that can see a level shift; it is a smoke test, not a crisis study.

## 8. Status of computation

Unit tests: no future edges; AUPRC in (0,1) on a hand example; tensor
shapes. A toy run writes `results/tables/toy_auprc.csv` and a hero
heatmap of returns plus the t=80 correlation graph, titled as TOY
connectedness. **NO FULL RESULTS YET** on Nasdaq-100, COVID, or SVB.
Do not put the toy AUPRC into a paper abstract.

## 9. Identification limits (for the discussant)

Even with perfect no-future hygiene, the graph is a function of the
same returns that enter the node features. A GCN can then act as a
smoother of a common factor that is already in the LSTM's input. A
finding that GCN beats LSTM on AUPRC may be "the factor is easier to
read from the correlation matrix" rather than "connectedness forecasts
crashes". Pre-specified controls: (a) compare to a GCN on the sector
graph only (no return-based edges); (b) shuffle edges as a placebo
graph. Those two checks are not fully wired into `run_toy.py` yet;
they are listed here so they cannot be dropped after peeking.

Common shocks vs transmission: the toy DGP has a common jump. A model
that looks at average correlation will look good. That is a feature of
the smoke test and a bug of any crisis narrative built on it.

## 10. Planned tables (empty)

Table 1: AUPRC LSTM vs GCN vs temporal, Nasdaq-100, 2015-2019 train /
2020 test — empty. Table 2: same, 2022 train / 2023-03 test — empty.
Table 3: edge placebo — empty. Table 4: lead-lag vs corr vs sector
ablation — empty. Figure 1: COVID subgraph from public prices — not
drawn; Figure 2: TOY t=80 — drawn if `run_toy` succeeded.

## 11. Risks

Fifteen synthetic names overfit instantly. Dense N-by-N GCN does not
scale to the full Russell 3000 without sparsity. Thresholded
correlation is noisy at lookback 20. AUPRC variance is high with few
positives; we will need bootstrap CIs on real data. Yahoo-style URLs
are unstable; CI must stay synthetic. Calling a 2020 subgraph "COVID
contagion" would be a terminology error even if prices download.

## 12. Workshop 12-minute arc

Slide 1: connectedness vs contagion, one sentence. Slide 2: no-future
graph construction. Slide 3: three models, AUPRC. Slide 4: empty
Nasdaq table, TOY figure with a red TOY stamp. Slide 5: placebo graph
as the next experiment. Closing: a GNN on a correlation matrix is a
smoother until a design says otherwise.

## 13. Implementation map

`src/graphs/build.py` — corr, lead-lag, sector, assert_no_future.
`src/models/` — lstm_independent, static_gcn, temporal.
`src/metrics/auprc.py` — labels and AUPRC.
`src/data_panel.py` — synthetic panel and failed Nasdaq hook.
`src/viz/hero.py` — figure.
`src/run_toy.py` — one-epoch smoke.
`tests/test_no_future.py` — the tests that must stay green.

## 14. Next actions

1. If a licensed daily panel appears, cache it as CSV and do not git
   the raw vendor file if the licence forbids.
2. Wire sector-only and shuffled-edge placebos into the train entry.
3. Add bootstrap AUPRC CIs.
4. Draw COVID/SVB subgraphs only from public prices, labelled as
   connectedness snapshots.
5. Keep causal language out of figure captions.

## 15. Conclusion of the outline

The software enforces a no-future-edge constraint and an AUPRC metric
for rare crash labels. It does not yet evaluate a real market. The
honest workshop sentence is that independent LSTMs, static GCNs, and a
small temporal GNN are wired, tested for leakage, and untested on
Nasdaq-100. A planted shock at t=80 is a unit-test-sized event, not a
crisis study, and not contagion.


## 16. Notes for a graph-ML lab

A graph-ML reader will notice the absence of a learned adjacency
(MTGNN-style) and of attention over neighbours (GAT). Those are
deliberate v0.1 omissions. Learned graphs on 15 TOY names will overfit
the planted shock and produce a pretty figure that does not travel.
Attention weights will be interpreted as "who infects whom" in a
seminar; we would then spend the Q&A unsaying contagion. The first
scientific deliverable is a leakage-free pipeline and an AUPRC
comparison against an independent LSTM. Adding GAT is a pre-registered
extension, not a default.

A second lab comment: EvolveGCN evolves parameters, we evolve states.
If a referee insists on the paper's module, we will implement
EvolveGCN-O as a config flag rather than rename our GRU and hope no
one reads Pareja et al. Until then the model name in code is
`TemporalGRUGcn`.

Finally, Diebold-Yilmaz FEVD graphs require a VAR. A 15-variable VAR
on 20-day windows is unidentified in the usual sense; that is why
lead-lag here is a lagged correlation, disclosed as such, and why we
do not print a "spillover index" on the toy panel.


The CI workflow installs the CPU extra of torch and runs pytest without
network. That constraint is part of the scientific protocol: a test that
needs Yahoo is not a test. Replicators should treat a green CI as
evidence that leakage checks pass, not as evidence that crash AUPRC on
a 15-name toy panel will match a Nasdaq-100 exercise in 2020 or 2023.

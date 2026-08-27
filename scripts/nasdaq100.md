# Nasdaq-100 prices (optional, public)

This project does not scrape. If you have a public daily price panel
(e.g. obtained via a licensed vendor API or a manual download that you
are allowed to store), place it at `data/nasdaq100_prices.csv` with
dates as rows and tickers as columns of returns.

`src/data_panel.py:try_nasdaq100_prices` attempts a short timeout fetch
of a public QQQ history URL and **fails gracefully** to the synthetic
15-name panel. CI must not depend on Yahoo.

COVID-19 (2020-03) and SVB (2023-03) subgraphs are intended later as
*illustrative connectedness snapshots*, not as causal contagion proofs.
Until public prices exist, the hero figure is a TOY shock at t=80.

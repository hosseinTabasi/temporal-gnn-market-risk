"""Synthetic 15-name panel and a Nasdaq-100 download that fails gracefully."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from graphs.build import TICKERS_TOY, SECTOR

def synthetic_panel(n_steps: int = 120, n: int = 15, seed: int = 0,
                    shock_t: int = 80) -> dict:
    """Factor panel plus a labelled TOY shock at shock_t.

    Common factor + sector factors + idiosyncratic. At shock_t a
    negative common jump hits sector 0 harder. Crash labels are left-tail
    returns. This is not COVID, not SVB, and not contagion.
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, n_steps)
    sec = rng.normal(0, 0.008, size=(n_steps, 3))
    eps = rng.normal(0, 0.012, size=(n_steps, n))
    r = np.zeros((n_steps, n))
    for i in range(n):
        s = i // 5
        r[:, i] = 0.8 * f + 0.5 * sec[:, s] + eps[:, i]
    st = min(shock_t, n_steps - 1)
    r[st, :5] -= 0.08
    r[st, 5:] -= 0.03
    return {
        "returns": r.astype(np.float32),
        "tickers": TICKERS_TOY[:n],
        "shock_t": shock_t,
        "source": "synthetic_toy",
        "label": "TOY",
    }

def try_nasdaq100_prices(root: Path | str = ".", timeout: float = 4.0):
    """Best-effort public download. Returns None on any failure.

    Intended for a later COVID/SVB subgraph from public prices. We do
    not scrape a vendor. Failure is the expected CI path.
    """
    root = Path(root)
    cache = root / "data" / "nasdaq100_prices.csv"
    if cache.is_file():
        df = pd.read_csv(cache, index_col=0)
        return df, "cache"
    try:
        import urllib.request
        url = "https://query1.finance.yahoo.com/v7/finance/download/QQQ?interval=1d&events=history"
        urllib.request.urlopen(url, timeout=timeout).read()
        return None, "download_unparsed"
    except Exception:
        return None, "synthetic_toy"

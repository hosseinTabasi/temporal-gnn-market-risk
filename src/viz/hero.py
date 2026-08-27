"""Hero figure: TOY shock subgraph at t=80 (or public prices if present)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def write_hero(returns: np.ndarray, adj: np.ndarray, t: int, out: Path,
               title: str = "TOY shock connectedness (not contagion)") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        np.savez(out.with_suffix(".npz"), adj=adj, t=np.array(t))
        return out.with_suffix(".npz")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].imshow(returns.T, aspect="auto", cmap="coolwarm", vmin=-0.05, vmax=0.05)
    ax[0].axvline(t, color="k", lw=0.8)
    ax[0].set_title("returns (names x time)")
    ax[1].imshow(adj, cmap="Greys")
    ax[1].set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

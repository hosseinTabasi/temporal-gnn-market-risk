"""AUPRC for rare crash labels. Accuracy is the wrong headline metric."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def crash_labels(returns: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Binary crash: asset-day in the left q tail of the pooled return dist.

    returns: (T, N). labels: (T, N) in {0,1}. Rare by construction.
    """
    thr = np.quantile(returns, q)
    return (returns <= thr).astype(np.int64)

def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """sklearn average precision; defined even if the positive class is rare.

    If there is no positive in y_true, sklearn returns 0.0 in recent
    versions when pos_label=1 is missing; we return nan and document it.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

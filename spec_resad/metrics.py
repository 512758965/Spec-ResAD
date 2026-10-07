"""Image-level and pixel-level AUROC."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    scores = np.asarray(scores).astype(np.float64).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

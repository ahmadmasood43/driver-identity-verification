from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    clusters: ArrayLike,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap that resamples whole clusters.

    `statistic` gets an array of sample indices (with repeats) and returns a scalar.
    Pass subject IDs as `clusters` so all samples from one subject move together;
    pass np.arange(n) for an ordinary per-sample bootstrap.
    """
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        picks = rng.integers(0, len(members), len(members))
        stats[b] = statistic(np.concatenate([members[p] for p in picks]))

    tail = (1 - confidence) / 2
    lo, hi = np.quantile(stats, [tail, 1 - tail])
    return CI(float(statistic(np.arange(inverse.size))), float(lo), float(hi))

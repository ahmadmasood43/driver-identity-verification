"""Seeded, subject-disjoint splits. Split files are committed before any model sees the data."""

import json
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path

import numpy as np

DEFAULT_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}


def assign(
    subjects: Iterable[str],
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    *,
    seed: int,
    strata: Mapping[str, Hashable] | None = None,
) -> dict[str, list[str]]:
    """Split subject IDs by `fractions`. With `strata`, each stratum is split separately so
    every demographic group keeps roughly the same proportions in every split."""
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("fractions must sum to 1")
    ids = sorted(set(subjects))
    groups: dict[Hashable, list[str]] = defaultdict(list)
    for s in ids:
        groups[strata[s] if strata else None].append(s)

    rng = np.random.default_rng(seed)
    names = list(fractions)
    cum = np.cumsum([fractions[n] for n in names])
    out: dict[str, list[str]] = {n: [] for n in names}
    # Sorted stratum order keeps the result independent of dict insertion order.
    for key in sorted(groups, key=repr):
        members = [groups[key][i] for i in rng.permutation(len(groups[key]))]
        cuts = np.round(cum * len(members)).astype(int)
        start = 0
        for name, end in zip(names, cuts, strict=True):
            out[name].extend(members[start:end])
            start = end
    return {n: sorted(v) for n, v in out.items()}


def check_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for s in ids:
            if s in seen:
                raise ValueError(f"subject {s} in both {seen[s]} and {name}")
            seen[s] = name


def save(path: Path, splits: Mapping[str, list[str]], *, seed: int, **meta) -> None:
    check_disjoint(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"seed": seed, **meta, "splits": {k: sorted(v) for k, v in splits.items()}}
    path.write_text(json.dumps(doc, indent=1) + "\n")


def load(path: Path) -> dict[str, list[str]]:
    splits = json.loads(path.read_text())["splits"]
    check_disjoint(splits)
    return splits

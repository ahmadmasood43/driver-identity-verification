import random
from collections import Counter

import pytest

from idv.data import splits


def subjects(n):
    return [f"s{i:04d}" for i in range(n)]


def test_disjoint_and_complete():
    ids = subjects(1000)
    out = splits.assign(ids, seed=0)
    splits.check_disjoint(out)
    assert sorted(s for v in out.values() for s in v) == ids


def test_fractions():
    out = splits.assign(subjects(1000), seed=0)
    assert {k: len(v) for k, v in out.items()} == {"train": 600, "dev": 200, "test": 200}


def test_same_seed_same_split_regardless_of_input_order():
    ids = subjects(300)
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    assert splits.assign(ids, seed=3) == splits.assign(shuffled, seed=3)
    assert splits.assign(ids, seed=3) != splits.assign(ids, seed=4)


def test_stratified_keeps_group_proportions():
    ids = subjects(1000)
    # 10% of subjects in a minority group.
    strata = {s: ("V-VI" if i % 10 == 0 else "I-IV") for i, s in enumerate(ids)}
    out = splits.assign(ids, seed=0, strata=strata)
    for name, frac in splits.DEFAULT_FRACTIONS.items():
        counts = Counter(strata[s] for s in out[name])
        assert counts["V-VI"] == pytest.approx(100 * frac, abs=1)


def test_check_disjoint_catches_leak():
    with pytest.raises(ValueError, match="s1"):
        splits.check_disjoint({"train": ["s0", "s1"], "test": ["s1"]})


def test_save_load_roundtrip(tmp_path):
    out = splits.assign(subjects(50), seed=7)
    p = tmp_path / "cc.json"
    splits.save(p, out, seed=7, dataset="casual-conversations-v1")
    assert splits.load(p) == out


def test_bad_fractions():
    with pytest.raises(ValueError):
        splits.assign(subjects(10), {"train": 0.5, "test": 0.4}, seed=0)

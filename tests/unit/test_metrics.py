import math

import numpy as np
import pytest

from idv.eval.metrics import (
    bpcer_at_apcer,
    eer,
    error_curve,
    far,
    frr,
    pad_errors,
    threshold_at_far,
)


def test_far_frr_accept_at_threshold():
    assert far([0.1, 0.5, 0.9], 0.5) == pytest.approx(2 / 3)
    assert frr([0.1, 0.5, 0.9], 0.5) == pytest.approx(1 / 3)


def test_empty_scores_rejected():
    with pytest.raises(ValueError):
        far([], 0.5)


@pytest.mark.parametrize("target", [0.0, 0.001, 0.01, 0.07, 0.29, 0.5, 1.0])
def test_threshold_at_far_is_tight(target):
    rng = np.random.default_rng(0)
    imp = rng.normal(size=1000)
    t = threshold_at_far(imp, target)
    assert far(imp, t) <= target
    if t > -math.inf:
        # One step lower must break the target, otherwise t wasn't the lowest.
        assert far(imp, np.nextafter(t, -math.inf)) > target


def test_threshold_at_far_with_ties_stays_under_target():
    imp = [0.5] * 10 + [0.1] * 90
    t = threshold_at_far(imp, 0.05)
    assert t > 0.5
    assert far(imp, t) == 0.0


def test_error_curve_endpoints():
    t, fa, fr = error_curve([0.6, 0.8, 0.9], [0.1, 0.2, 0.7])
    assert np.all(np.diff(t) > 0)
    assert fa[0] == 1.0 and fr[0] == 0.0
    assert np.all(np.diff(fa) <= 0) and np.all(np.diff(fr) >= 0)


def test_eer_separable_is_zero():
    rate, t = eer([0.8, 0.9], [0.1, 0.2])
    assert rate == 0.0
    assert 0.2 < t <= 0.8


def test_eer_symmetric_overlap():
    rate, _ = eer([0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6])
    assert rate == pytest.approx(0.25)


def test_pad_errors_uses_worst_species():
    bona = [0.9, 0.8, 0.3, 0.95]
    attack = [0.1, 0.2, 0.7, 0.6, 0.1]
    species = ["print", "print", "replay", "replay", "mask"]
    e = pad_errors(bona, attack, species, threshold=0.5)
    assert e.apcer_per_species == {"mask": 0.0, "print": 0.0, "replay": 1.0}
    assert e.apcer == 1.0
    assert e.bpcer == 0.25
    assert e.acer == pytest.approx(0.625)


def test_bpcer_at_apcer_meets_target_for_every_species():
    rng = np.random.default_rng(1)
    attack = np.concatenate([rng.normal(-2, 1, 500), rng.normal(0, 1, 500)])
    species = np.array(["print"] * 500 + ["replay"] * 500)
    bona = rng.normal(2, 1, 1000)
    bpcer, t = bpcer_at_apcer(bona, attack, species, target=0.01)
    e = pad_errors(bona, attack, species, t)
    assert e.apcer <= 0.01
    assert e.bpcer == bpcer
    # The harder species sets the threshold.
    assert e.apcer_per_species["print"] < e.apcer_per_species["replay"]

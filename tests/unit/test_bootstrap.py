import numpy as np

from idv.eval.bootstrap import bootstrap_ci


def test_point_and_ordering():
    x = np.random.default_rng(0).normal(size=300)
    ci = bootstrap_ci(lambda idx: x[idx].mean(), np.arange(x.size), n_resamples=500)
    assert ci.point == x.mean()
    assert ci.lo <= ci.point <= ci.hi


def test_deterministic_for_seed():
    x = np.random.default_rng(0).normal(size=100)
    a = bootstrap_ci(lambda idx: x[idx].mean(), np.arange(100), n_resamples=200, seed=7)
    b = bootstrap_ci(lambda idx: x[idx].mean(), np.arange(100), n_resamples=200, seed=7)
    assert a == b


def test_cluster_resampling_widens_ci_for_correlated_samples():
    # 40 subjects x 25 samples; most variance is between subjects.
    rng = np.random.default_rng(0)
    subject = np.repeat(np.arange(40), 25)
    x = rng.normal(size=40)[subject] + rng.normal(scale=0.1, size=subject.size)

    def mean(idx):
        return x[idx].mean()

    iid = bootstrap_ci(mean, np.arange(x.size), n_resamples=1000)
    clustered = bootstrap_ci(mean, subject, n_resamples=1000)
    assert (clustered.hi - clustered.lo) > 3 * (iid.hi - iid.lo)

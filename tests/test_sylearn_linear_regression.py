"""Tests for syLinearRegression: fit(), partial_fit() and merge().

Run:  python tests/test_sylearn_linear_regression.py      (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sylearn import syLinearRegression  # noqa: E402

RNG = np.random.default_rng(0)
X = RNG.random((1000, 5)) * [300, 450, 120, 110, 25]          # AQI-like scales
Y = X @ [1.2, 0.25, 0.1, 0.6, 10.0] + 7 + RNG.normal(0, 5, 1000)


def same(a, b):
    return (np.allclose(a.weights, b.weights, rtol=1e-6, atol=1e-8)
            and np.isclose(a.intercept_, b.intercept_, rtol=1e-6, atol=1e-6))


def test_fit_unchanged_matches_sklearn():
    from sklearn.linear_model import LinearRegression
    m = syLinearRegression()
    weights, intercept = m.fit(X, Y)                 # fit() still returns (weights, intercept)
    sk = LinearRegression().fit(X, Y)
    assert np.allclose(weights, sk.coef_) and np.isclose(intercept, sk.intercept_)
    assert np.allclose(m.predict(X), sk.predict(X))


def test_partial_fit_in_batches_equals_full_fit():
    full = syLinearRegression(); full.fit(X, Y)
    inc = syLinearRegression()
    for i in range(0, len(X), 137):                  # uneven batches on purpose
        inc.partial_fit(X[i:i + 137], Y[i:i + 137])
    assert same(full, inc)
    assert inc.n_seen == len(X)


def test_partial_fit_one_row_at_a_time():
    full = syLinearRegression(); full.fit(X[:200], Y[:200])
    inc = syLinearRegression()
    for i in range(200):
        inc.partial_fit(X[i:i + 1], Y[i:i + 1])
    assert same(full, inc)


def test_continue_after_fit():
    full = syLinearRegression(); full.fit(X, Y)
    m = syLinearRegression(); m.fit(X[:600], Y[:600])
    m.partial_fit(X[600:], Y[600:])
    assert same(full, m)


def test_merge_equals_training_on_all_data():
    full = syLinearRegression(); full.fit(X, Y)
    parts = [syLinearRegression().partial_fit(X[i:i + 250], Y[i:i + 250]) for i in range(0, 1000, 250)]
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p)
    assert same(full, merged)
    assert merged.n_seen == 1000
    # merging does not change the original models
    assert parts[0].n_seen == 250


def test_bad_input_raises_an_error():
    m = syLinearRegression().partial_fit(X[:10], Y[:10])
    for bad in [lambda: m.partial_fit(X[:5, :3], Y[:5]),          # wrong number of features
                lambda: m.partial_fit(X[:5], Y[:4]),               # X and y lengths differ
                lambda: syLinearRegression().merge(m)]:            # merging an untrained model
        try:
            bad()
            raise AssertionError("expected an error")
        except (ValueError, TypeError):
            pass


def test_old_pickled_model_still_predicts():
    """A model pickled by sylearn 0.1.0 has no running sums; predict() must still work."""
    m = syLinearRegression(); m.fit(X, Y)
    old = syLinearRegression.__new__(syLinearRegression)
    old.__dict__.update({"weights": m.weights, "intercept_": m.intercept_})
    assert np.allclose(old.predict(X), m.predict(X))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} tests passed.")

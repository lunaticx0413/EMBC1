import numpy as np
import pandas as pd
import pytest

from embc.design import build_design, fit, intercept_kernel, residualize
from embc.stream import design_inputs


def _one_segment(n):
    return np.zeros(n, int)


def test_fir_rows_and_values():
    X, cols = build_design(50, _one_segment(50), {"fix": {"onset": np.array([10])}}, lags={"fix": (-2, 3)})
    D = X.toarray()
    assert len(cols) == 6 and cols[0] == ("fix", "intercept", -2)
    for j, lag in enumerate(range(-2, 4)):
        assert D[10 + lag, j] == 1 and D[:, j].sum() == 1


def test_fir_does_not_cross_segment_boundary():
    seg = np.repeat([0, 1], 20)
    X, _ = build_design(40, seg, {"fix": {"onset": np.array([18])}}, lags={"fix": (0, 5)})
    assert np.flatnonzero(X.toarray().sum(1)).tolist() == [18, 19]


@pytest.mark.parametrize("events,cont", [
    ({"fix": {"onset": np.array([1]), "zipf": np.array([1.0])}}, None),
    ({"fix": {"onset": np.array([1]), "word_length": np.array([1.0])}}, None),
    ({"fix": {"onset": np.array([1]), "surprisal": np.array([1.0])}}, None),
    ({"lexical": {"onset": np.array([1])}}, None),
    ({"fix": {"onset": np.array([1])}}, {"nchar": np.ones(30)}),
])
def test_lexical_covariates_are_rejected(events, cont):
    with pytest.raises(ValueError):
        build_design(30, _one_segment(30), events, cont)


def _simulate(n=30000, seed=0, C=2):
    rng = np.random.default_rng(seed)
    onsets = np.cumsum(rng.integers(18, 34, n // 25))
    onsets = onsets[onsets < n - 100]
    dur = rng.integers(12, 28, len(onsets))
    amp = rng.standard_normal(len(onsets))
    lags_f, lags_s = np.arange(-20, 81), np.arange(-10, 41)
    kf = np.stack([np.sin(lags_f / (7 + c)) * np.exp(-((lags_f - 25) / 30.0) ** 2) for c in range(C)], 1)
    ka = np.stack([np.cos(lags_f / 5.0) * np.exp(-((lags_f - 10) / 15.0) ** 2) * (c + 1) for c in range(C)], 1)
    ks = np.stack([np.exp(-((lags_s - 3) / 4.0) ** 2) * (2 - c) for c in range(C)], 1)
    Y = np.zeros((n, C))
    for o, d, a in zip(onsets, dur, amp):
        Y[o + lags_f] += kf + a * ka
        Y[o + d + lags_s] += ks
    events = {"fix": {"onset": onsets, "sacc_in_amp": amp}, "sacc": {"onset": onsets + dur}}
    return Y, events, dict(kf=kf, ka=ka, ks=ks), rng


def test_recovers_overlapping_kernels():
    Y, events, true, rng = _simulate()
    Y = Y + 0.05 * rng.standard_normal(Y.shape)
    X, cols = build_design(len(Y), _one_segment(len(Y)), events)
    B, info = fit(X, Y, np.ones(len(Y), bool), ridge=1e-6)
    lags, K = intercept_kernel(B, cols)
    assert np.corrcoef(K.ravel(), true["kf"].ravel())[0, 1] > 0.99
    ka = B[[i for i, c in enumerate(cols) if c[:2] == ("fix", "sacc_in_amp")]]
    assert np.corrcoef(ka.ravel(), true["ka"].ravel())[0, 1] > 0.99
    assert np.isfinite(info["cond_design"])


def test_residualization_is_linear_and_keeps_fixation_intercept():
    Y, events, true, rng = _simulate(n=12000, seed=1)
    seg, valid = _one_segment(len(Y)), np.ones(len(Y), bool)
    Z = rng.standard_normal(Y.shape)
    r1 = residualize(Y, valid, seg, events, {})
    r2 = residualize(Z, valid, seg, events, {})
    r12 = residualize(Y + Z, valid, seg, events, {})
    np.testing.assert_allclose(r12["res"], r1["res"] + r2["res"], atol=1e-8)
    # a signal made only of the average fixation response survives in res and is removed in resfull
    only_int = np.zeros_like(Y)
    for o in events["fix"]["onset"]:
        only_int[o + np.arange(-20, 81)] += true["kf"]
    r = residualize(only_int, valid, seg, events, {}, ridge=1e-8)
    assert np.abs(r["res"] - only_int).max() < 1e-3 * np.abs(only_int).max()
    assert np.abs(r["resfull"]).max() < 1e-3 * np.abs(only_int).max()


def test_design_ignores_word_identity():
    rng = np.random.default_rng(0)
    n = 40
    fix = pd.DataFrame(dict(onset=np.arange(5, 5 + 20 * n, 20), offset=np.arange(15, 15 + 20 * n, 20),
                            x=rng.uniform(100, 800, n), y=rng.uniform(100, 400, n), sacc_in_amp=rng.uniform(0, 80, n),
                            sacc_in_dx=rng.normal(0, 40, n), sacc_in_dy=rng.normal(0, 5, n),
                            sacc_out_amp=rng.uniform(0, 80, n), sacc_out_dx=rng.normal(0, 40, n),
                            sacc_out_dy=rng.normal(0, 5, n), gap_in_ms=rng.uniform(10, 200, n),
                            word=[f"w{i}" for i in range(n)], type=[f"w{i}" for i in range(n)],
                            nchar=rng.integers(2, 12, n), zipf=rng.uniform(1, 7, n)))
    gaze = rng.standard_normal((900, 2))
    valid = np.ones(900, bool)
    X1, _ = build_design(900, _one_segment(900), *design_inputs(fix, gaze, valid))
    shuffled = fix.assign(word=fix.word.sample(frac=1, random_state=1).to_numpy(), nchar=0, zipf=0.0, type="x")
    X2, _ = build_design(900, _one_segment(900), *design_inputs(shuffled, gaze, valid))
    assert (X1 != X2).nnz == 0

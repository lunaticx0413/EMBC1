import numpy as np
import pytest

from embc import metrics as mt


def test_tie_aware_ranks_hand_computed():
    S = np.array([[0.5, 0.9, 0.5, 0.1], [0.9, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
    g, e = mt.gallery_ranks(S)
    assert g.tolist() == [1, 0, 3] and e.tolist() == [1, 0, 0]
    m = mt.rank_metrics(g, e, 4, ks=(1, 2))
    assert m["RR"] == pytest.approx([(1 / 2 + 1 / 3) / 2, 1.0, 1 / 4])
    assert m["R@1"] == pytest.approx([0.0, 1.0, 0.0])
    assert m["R@2"] == pytest.approx([0.5, 1.0, 0.0])
    assert m["nDCG"][0] == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / 2)
    assert m["rank"].tolist() == [2.5, 1.0, 4.0]


def test_all_tied_equals_chance():
    N = 100
    S = np.zeros((3, N))
    m = mt.rank_metrics(*mt.gallery_ranks(S), N)
    c = mt.chance(N)
    for k in ("R@1", "R@5", "R@10", "RR", "nDCG"):
        assert m[k] == pytest.approx([c[k]] * 3)
    assert c["R@5"] == pytest.approx(0.05)


def test_random_scores_match_chance():
    rng = np.random.default_rng(0)
    N = 100
    m = mt.rank_metrics(*mt.gallery_ranks(rng.standard_normal((40000, N))), N)
    c = mt.chance(N)
    assert m["RR"].mean() == pytest.approx(c["RR"], abs=0.003)
    assert m["R@5"].mean() == pytest.approx(0.05, abs=0.004)


def test_sign_flip_and_bootstrap():
    assert mt.sign_flip_p(np.zeros(8)) == 1.0
    assert mt.sign_flip_p(np.ones(10)) < 0.01
    m, lo, hi = mt.bootstrap_ci(np.arange(10.0))
    assert lo < m < hi and m == pytest.approx(4.5)


def test_score_galleries_cosine():
    T = np.eye(4, dtype=np.float32)
    Z = np.array([[2.0, 0, 0, 0], [0, 0, 1, 1]])
    G = np.array([[0, 1, 2], [2, 3, 0]])
    S = mt.score_galleries(Z, T, G)
    np.testing.assert_allclose(S, [[1, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0]], atol=1e-6)

import numpy as np
import pytest

from embc import lexicon as lx
from embc.inputs import LAGS, add_kernel, baseline, subject_shift, window_index


def test_clean_and_content():
    stop = lx.stopwords()
    assert lx.clean_word("Ford,") == "ford"
    assert lx.clean_word("'borrowed'") == "borrowed"
    assert lx.clean_word("(born") == "born"
    assert not lx.is_content("1936", "NOUN", stop)
    assert not lx.is_content("the", "NOUN", stop)
    assert lx.is_content("career-ending", "ADJ", stop)
    assert not lx.is_content("ford", "PROPN", stop)


def test_ptb_mapping():
    assert [lx.ptb_to_upos(t) for t in ("NN", "NNPS", "VBD", "JJS", "RB", "DT", "MD")] == \
        ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "OTHER", "AUX"]


def _toy(n_types=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_types, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    attrs = dict(upos=rng.choice(np.array(["NOUN", "VERB", "ADJ", "ADV"]), n_types),
                 nchar=rng.integers(2, 14, n_types).astype(float), zipf=rng.uniform(1, 7, n_types),
                 logtc=np.log1p(rng.poisson(3, n_types)).astype(float))
    attrs.update({f"{k}_sd": float(attrs[k].std()) for k in lx.MATCH_ATTRS})
    return emb, attrs


@pytest.mark.parametrize("kind", ["random", "matched", "balanced"])
def test_gallery_invariants(kind):
    emb, attrs = _toy()
    emb[7] = emb[3]                                         # type 7 is a near-duplicate of target 3
    pool = np.arange(400)
    G, info = lx.build_gallery(3, pool, attrs, emb, 50, kind, np.random.default_rng(0))
    assert len(G) == 50 and G[0] == 3 and 3 not in G[1:] and len(set(G)) == 50
    assert 7 not in G                                       # false negative excluded
    G2, _ = lx.build_gallery(3, pool, attrs, emb, 50, kind, np.random.default_rng(0))
    assert (G == G2).all()
    if kind == "matched":
        assert (attrs["upos"][G[1:]] == attrs["upos"][3]).all() and info["pos_fallback"] == 0
        Gr, _ = lx.build_gallery(3, pool, attrs, emb, 50, "random", np.random.default_rng(0))
        d = lambda g: np.abs(attrs["zipf"][g[1:]] - attrs["zipf"][3]).mean()  # noqa: E731
        assert d(G) < d(Gr)


def test_gallery_pool_too_small():
    emb, attrs = _toy(n_types=20)
    with pytest.raises(ValueError):
        lx.build_gallery(0, np.arange(20), attrs, emb, 50, "random", np.random.default_rng(0))


def test_window_and_baseline():
    bad = np.zeros(200, bool)
    bad[105] = True
    seg_id = np.repeat([0, 1], 100)
    c, valid = window_index(np.array([30, 95]), bad, seg_id)
    assert valid[0].all()                                   # 20 .. 89 stays inside sentence 0
    assert not valid[1, LAGS >= 5].any() and valid[1, LAGS < 5].all()
    X = np.ones((2, len(LAGS), 1)) * 3.0
    Xb = baseline(X, valid)
    assert np.abs(Xb[0]).max() == 0 and (Xb[1][~valid[1]] == 0).all()


def test_add_kernel_and_shift():
    kl = np.arange(-20, 81)
    K = kl[:, None].astype(float)
    X = add_kernel(np.zeros((1, len(LAGS), 1)), kl, K)
    assert X[0, :, 0].tolist() == LAGS.tolist()
    s = subject_shift("YAC", 50000)
    assert 1000 <= s < 49000 and s == subject_shift("YAC", 50000)


def test_balanced_gallery_is_frequency_prior_neutral():
    emb, attrs = _toy(n_types=600, seed=3)
    attrs["upos"][:] = "NOUN"
    pool = np.arange(600)
    t = int(np.argsort(attrs["logtc"])[300])                  # a mid-frequency target
    G, info = lx.build_gallery(t, pool, attrs, emb, 51, "balanced", np.random.default_rng(0))
    lt = attrs["logtc"]
    assert (lt[G[1:]] > lt[t]).sum() == (lt[G[1:]] < lt[t]).sum() and info["imbalance"] == 0

"""Gallery retrieval metrics (tie-aware), chance levels, subject-level statistics.

Gallery score matrices have the target in column 0. Ties are broken uniformly at random in expectation:
with g candidates scoring above the target and e tied with it, the rank is uniform on g+1 .. g+e+1.
"""
import itertools

import numpy as np

KS = (1, 5, 10)


def _cums(N):
    r = np.arange(1, N + 1, dtype=float)
    return np.concatenate([[0], np.cumsum(1 / r)]), np.concatenate([[0], np.cumsum(1 / np.log2(r + 1))])


def gallery_ranks(S):
    """S (n, N) scores, target in column 0 -> (greater count g, tie count e)."""
    s0 = S[:, :1]
    return (S[:, 1:] > s0).sum(1), (S[:, 1:] == s0).sum(1)


def rank_metrics(g, e, N, ks=KS):
    """Expected per-event metrics under random tie-breaking. Returns dict of (n,) arrays."""
    g, e = np.asarray(g), np.asarray(e)
    c1, c2 = _cums(N)
    out = {f"R@{k}": np.clip((k - g) / (e + 1), 0.0, 1.0) for k in ks}
    out["RR"] = (c1[g + e + 1] - c1[g]) / (e + 1)
    out["nDCG"] = (c2[g + e + 1] - c2[g]) / (e + 1)
    out["rank"] = g + (e + 2) / 2.0
    return out


def chance(N, ks=KS):
    c1, c2 = _cums(N)
    d = {f"R@{k}": k / N for k in ks}
    d.update(RR=c1[N] / N, nDCG=c2[N] / N, rank=(N + 1) / 2.0)
    return d


def score_galleries(Z, T, G):
    """Cosine scores of each event embedding against its gallery. Z (n,d), T (types,d) unit rows, G (n,N)."""
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    out = np.empty(G.shape, np.float32)
    for i in range(0, len(G), 2048):
        out[i:i + 2048] = np.einsum("nd,nkd->nk", Z[i:i + 2048], T[G[i:i + 2048]])
    return out


# ---------------------------------------------------------------- statistics over subjects

def subject_means(values, subjects):
    subjects = np.asarray(subjects)
    u = np.unique(subjects)
    return u, np.array([values[subjects == s].mean() for s in u])


def bootstrap_ci(x, n_boot=5000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def sign_flip_p(d, n_mc=20000, seed=0):
    """Two-sided paired sign-flip test on per-subject differences (exact for <= 16 subjects)."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], (n_mc, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((1 + (null >= obs - 1e-15).sum()) / (1 + len(null)))


def paired_contrast(a, b, subjects, n_boot=5000, seed=0):
    """Per-event metric arrays a, b over the same events -> subject-level paired difference a - b."""
    _, ma = subject_means(np.asarray(a), subjects)
    _, mb = subject_means(np.asarray(b), subjects)
    d = ma - mb
    m, lo, hi = bootstrap_ci(d, n_boot, seed)
    return dict(diff=m, lo=lo, hi=hi, p_signflip=sign_flip_p(d, seed=seed), n_subjects=int(len(d)))


def permutation_null(Z, T, G, subjects, metric="RR", n_perm=200, seed=0):
    """Null of the subject-mean metric when EEG embeddings are permuted across events of the same subject:
    breaks the EEG-word pairing, keeps each subject's embedding distribution and every gallery.
    Returns (n_perm, n_subjects) subject means (subjects in np.unique order)."""
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    N = G.shape[1]
    groups = [np.flatnonzero(subjects == s) for s in np.unique(subjects)]
    out = np.empty((n_perm, len(groups)))
    for p in range(n_perm):
        perm = np.arange(len(Z))
        for idx in groups:
            perm[idx] = idx[rng.permutation(len(idx))]
        g, e = gallery_ranks(score_galleries(Z[perm], T, G))
        v = rank_metrics(g, e, N)[metric]
        out[p] = [v[idx].mean() for idx in groups]
    return out

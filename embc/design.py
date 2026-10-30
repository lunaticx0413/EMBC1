"""Text-free oculomotor residualization by time-expanded (FIR) regression on continuous EEG.

Model (per subject, all channels share one design):  Y = X_ocul beta + e, fitted by ridge on valid samples.

  fix   events at fixation onsets:  intercept, incoming saccade amplitude/dx/dy, fixation x/y
  sacc  events at fixation offsets (outgoing saccade onset): intercept, outgoing amplitude/dx/dy
  gap   events at inter-fixation gaps > GAP_MS (blink / off-word fixation proxy): intercept
  gaze  continuous interpolated gaze x/y (corneo-retinal dipole proxy; ZuCo ships no EOG channels)

Derived signals:
  res      = Y - (all terms except the fixation-onset intercept)   ocular nuisance removed; the average
                                                                     fixation-related potential is kept
  resfull  = Y - (all terms)                                        also removes every fixation's average FRP
  res_ovl epoch_k = resfull[t_k + lags] + beta_fix_intercept(lags)  neighbours' FRP overlap removed, own kept

Only names in OCULAR_MODULATORS / CONTINUOUS may enter the design. Lexical covariates (word length,
frequency, surprisal, POS, identity, embeddings) are rejected: they belong to the separate diagnostic
rERP path and must never shape decoder inputs.
"""
import numpy as np
import scipy.sparse as sp

OCULAR_MODULATORS = {
    "fix": ("intercept", "sacc_in_amp", "sacc_in_dx", "sacc_in_dy", "fix_x", "fix_y"),
    "sacc": ("intercept", "sacc_out_amp", "sacc_out_dx", "sacc_out_dy"),
    "gap": ("intercept",),
}
CONTINUOUS = ("gaze_x", "gaze_y")
LAGS = {"fix": (-20, 80), "sacc": (-10, 40), "gap": (-10, 50)}     # samples at 100 Hz, inclusive
LEXICAL_FORBIDDEN = ("word", "type", "nchar", "length", "zipf", "freq", "surprisal", "predictab", "pos",
                     "embed", "semantic", "lemma")
KEEP = ("fix", "intercept")                                         # term kept in `res`


def check_names(events, continuous):
    for ev, mods in events.items():
        if ev not in OCULAR_MODULATORS:
            raise ValueError(f"event type {ev!r} is not an oculomotor event")
        for m in mods:
            low = m.lower()
            if any(tok in low for tok in LEXICAL_FORBIDDEN) and m not in OCULAR_MODULATORS[ev]:
                raise ValueError(f"lexical covariate {m!r} may not enter the residualization design")
            if m not in OCULAR_MODULATORS[ev]:
                raise ValueError(f"{m!r} is not a whitelisted modulator for {ev!r} events")
    for c in continuous:
        if c not in CONTINUOUS:
            raise ValueError(f"continuous regressor {c!r} is not whitelisted")


def build_design(n_samples, seg_id, events, continuous=None, lags=LAGS):
    """Sparse time-expanded design.

    n_samples   length of the concatenated stream
    seg_id      (n_samples,) int segment (sentence) id; FIR rows never cross a segment boundary
    events      {event_type: {"onset": int array, modulator_name: float array, ...}}
    continuous  {name: (n_samples,) float array}
    Returns (X csr (n_samples, P), columns list of (event_type, modulator, lag) / ("cont", name, 0)).
    """
    continuous = continuous or {}
    check_names({k: [m for m in v if m != "onset"] for k, v in events.items()}, list(continuous))
    rows, cols, vals, columns = [], [], [], []
    for ev in sorted(events):
        spec = events[ev]
        onset = np.asarray(spec["onset"], int)
        lo, hi = lags[ev]
        L = np.arange(lo, hi + 1)
        for m in OCULAR_MODULATORS[ev]:
            if m not in spec and m != "intercept":
                continue
            v = np.ones(len(onset)) if m == "intercept" else np.nan_to_num(np.asarray(spec[m], float))
            base = len(columns)
            columns.extend((ev, m, int(l)) for l in L)
            r = onset[:, None] + L[None, :]
            c = base + np.broadcast_to(np.arange(len(L)), r.shape)
            ok = (r >= 0) & (r < n_samples)
            ok[ok] &= seg_id[r[ok]] == np.broadcast_to(seg_id[np.clip(onset, 0, n_samples - 1)][:, None], r.shape)[ok]
            vv = np.broadcast_to(v[:, None], r.shape)
            keep = ok & (vv != 0)
            rows.append(r[keep]); cols.append(c[keep]); vals.append(vv[keep])
    for name in sorted(continuous):
        base = len(columns)
        columns.append(("cont", name, 0))
        v = np.nan_to_num(np.asarray(continuous[name], float))
        nz = np.flatnonzero(v)
        rows.append(nz); cols.append(np.full(len(nz), base)); vals.append(v[nz])
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_samples, len(columns)))
    return X, columns


def fit(X, Y, valid, ridge=1e-3):
    """Ridge fit on valid rows. ridge is relative to the mean diagonal of X'X. Returns beta (P, C), info."""
    Xv = X[np.flatnonzero(valid)]
    G = (Xv.T @ Xv).toarray()
    d = np.diag(G).copy()
    used = d > 0
    lam = ridge * d[used].mean()
    Gu = G[np.ix_(used, used)] + lam * np.eye(used.sum())
    B = np.zeros((X.shape[1], Y.shape[1]))
    rhs = Xv.T @ np.asarray(Y[valid], np.float64)
    B[used] = np.linalg.solve(Gu, rhs[used])
    s = 1.0 / np.sqrt(d[used])
    ev = np.linalg.eigvalsh(G[np.ix_(used, used)] * s[:, None] * s[None, :])   # unregularized, unit-diagonal
    info = dict(n_rows=int(valid.sum()), n_cols=int(X.shape[1]), n_cols_used=int(used.sum()), lam=float(lam),
                cond_design=float(np.sqrt(ev.max() / max(ev.min(), 1e-300))))   # cond(X), unit-norm columns
    return B, info


def predict_parts(X, B, columns):
    """Return (kept, nuisance): the fixation-intercept prediction and everything else."""
    keep = np.array([c[0] == KEEP[0] and c[1] == KEEP[1] for c in columns])
    kept = X[:, keep] @ B[keep]
    nuis = X[:, ~keep] @ B[~keep]
    return np.asarray(kept), np.asarray(nuis)


def intercept_kernel(B, columns, event="fix"):
    """beta of the event intercept over its lags: (lags array, (L, C) kernel)."""
    idx = [i for i, c in enumerate(columns) if c[0] == event and c[1] == "intercept"]
    return np.array([columns[i][2] for i in idx]), B[idx]


def residualize(Y, valid, seg_id, events, continuous, ridge=1e-3):
    """Full pipeline on one subject stream. Returns dict(res, resfull, kernel_lags, kernel, info)."""
    X, columns = build_design(len(Y), seg_id, events, continuous)
    B, info = fit(X, Y, valid, ridge)
    kept, nuis = predict_parts(X, B, columns)
    res = np.asarray(Y, np.float64) - nuis
    resfull = res - kept
    res[~valid] = 0.0
    resfull[~valid] = 0.0
    lags, K = intercept_kernel(B, columns)
    yv = np.asarray(Y, np.float64)[valid]
    info.update(var_raw=float(yv.var(0).mean()), var_res=float(res[valid].var(0).mean()),
                var_resfull=float(resfull[valid].var(0).mean()), var_nuisance=float(nuis[valid].var(0).mean()))
    return dict(res=res, resfull=resfull, kernel_lags=lags, kernel=K, info=info, X=X, columns=columns, B=B)

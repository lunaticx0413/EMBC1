"""Decoder inputs per condition. Every EEG condition uses the same -100..+600 ms window and baseline.

raw        ZuCo-preprocessed EEG (already Automagic EOG-regressed + MARA ICA-cleaned)
res        oculomotor-residualized EEG (ocular nuisance removed, average FRP kept)
res_ovl    res with neighbouring fixations' average FRP removed (own fixation's average FRP kept)
ocfit      raw - res: the EEG part explained by the oculomotor model (EOG-like control; no EOG channels)
*_shift    the same signal epoched at circularly shifted onsets (fixed >= 10 s shift per subject):
           keeps the signal's distribution and spectrum, breaks the EEG <-> word time correspondence
noise      i.i.d. N(0, 1) inputs of the same shape
gaze       vector of oculomotor measures for the event (no EEG)
"""
import zlib

import numpy as np
import pandas as pd

from . import paths
from .events import POST, PRE

EEG_CONDITIONS = ("raw", "res", "res_ovl", "ocfit", "raw_shift", "res_shift", "noise")
GAZE_FEATURES = ("log_ffd", "log_gd", "log_trt", "nfix_ours", "x", "y", "sacc_in_dx", "sacc_in_dy", "sacc_in_amp",
                 "sacc_out_dx", "sacc_out_dy", "sacc_out_amp", "log_gap_in", "log_gap_out")
MIN_SHIFT = 1000                                   # samples (10 s)


def channel_keep():
    cp = pd.read_csv(paths.CHANPOS)
    return np.flatnonzero(cp.is_constant.to_numpy() == 0)


def subject_shift(subject, n, seed=0):
    rng = np.random.default_rng([seed, zlib.crc32(subject.encode())])
    return int(rng.integers(MIN_SHIFT, n - MIN_SHIFT))


def _signal(z, base):
    if base == "raw":
        return z["raw"]
    if base == "res":
        return z["res"]
    if base == "ocfit":
        return (z["raw"].astype(np.float32) - z["res"].astype(np.float32))
    if base == "res_ovl":
        return z["resfull"]
    raise ValueError(base)


LAGS = np.arange(-PRE, POST)


def window_index(onsets, bad, seg_id):
    """Sample indices (n, T) of each epoch window and its valid mask (inside the onset's sentence, not bad)."""
    n_samp = len(bad)
    onsets = np.asarray(onsets, int)
    idx = onsets[:, None] + LAGS[None, :]
    c = np.clip(idx, 0, n_samp - 1)
    valid = ((idx >= 0) & (idx < n_samp) & ~bad[c]
             & (seg_id[c] == seg_id[np.clip(onsets, 0, n_samp - 1)][:, None]))
    return c, valid


def baseline(X, valid):
    """X (n, T, C): subtract the mean of valid -100..0 ms samples; zero invalid samples."""
    X = np.where(valid[:, :, None], X, 0.0)
    b = X[:, :PRE].sum(1) / np.maximum(valid[:, :PRE].sum(1), 1)[:, None]
    return np.where(valid[:, :, None], X - b[:, None, :], 0.0)


def add_kernel(X, kernel_lags, kernel):
    """Add an event's own average response (kernel over its lags) to (n, T, C) epochs: res_ovl."""
    return X + kernel[np.searchsorted(kernel_lags, LAGS)][None]


def subject_epochs(subject, onsets, cond, seed=0):
    """(n, C, T) float32 epochs and (n, T) bool valid mask for one subject's events."""
    keep = channel_keep()
    z = np.load(paths.continuous_path(subject))
    bad, seg = z["bad"], z["seg"]
    seg_id = np.repeat(np.arange(len(seg)), seg[:, 2])
    onsets = np.asarray(onsets, int)
    base = cond[:-6] if cond.endswith("_shift") else cond
    if cond.endswith("_shift"):
        onsets = (onsets + subject_shift(subject, len(bad), seed)) % len(bad)
    c, valid = window_index(onsets, bad, seg_id)
    X = np.asarray(_signal(z, base)[c.reshape(-1)][:, keep], np.float32).reshape(len(onsets), len(LAGS), len(keep))
    if base == "res_ovl":
        X = add_kernel(X, z["kernel_lags"], z["kernel"][:, keep])
    return baseline(X, valid).transpose(0, 2, 1).astype(np.float32), valid


def load_condition(cond, events, seed=0):
    """Inputs for the events DataFrame (columns subject, onset, event_id) in its row order.
    EEG conditions -> (X float16 (n, C, T), M bool (n, T)); gaze -> (X float32 (n, D), None)."""
    if cond == "gaze":
        return gaze_matrix(events), None
    n, C, T = len(events), len(channel_keep()), PRE + POST
    X = np.zeros((n, C, T), np.float16)
    M = np.zeros((n, T), bool)
    if cond == "noise":
        rng = np.random.default_rng(seed + 7919)
        for i in range(0, n, 4096):
            X[i:i + 4096] = rng.standard_normal((min(4096, n - i), C, T)).astype(np.float16)
        M[:] = True
        return X, M
    if cond not in EEG_CONDITIONS:
        raise ValueError(cond)
    subj = events.subject.to_numpy()
    for s in np.unique(subj):
        r = np.flatnonzero(subj == s)
        X[r], M[r] = subject_epochs(s, events.onset.to_numpy()[r], cond, seed)
    return X, M


def gaze_matrix(events):
    e = events
    cols = dict(log_ffd=np.log(e.ffd_ms), log_gd=np.log(e.gd_ms), log_trt=np.log(e.trt_ms),
                log_gap_in=np.log1p(e.gap_in_ms.clip(lower=0)), log_gap_out=np.log1p(e.gap_out_ms.clip(lower=0)))
    return np.stack([np.asarray(cols[f] if f in cols else e[f], float) for f in GAZE_FEATURES], 1).astype(np.float32)

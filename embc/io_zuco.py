"""ZuCo 2.0 NR reader: per-subject continuous 100 Hz EEG stream, fixation table and gaze trace.

Preprocessing matches the parent project's fixation epochs (projects/12 build_fixepochs.py): per sentence,
demean, NaN samples zeroed and masked, zero-phase 40 Hz low-pass, 500 -> 100 Hz resample_poly. The ZuCo
release is already Automagic-processed (0.5-50 Hz, EOG regression, MARA ICA), so this stream is the
"ZuCo-preprocessed" condition, not unprocessed EEG.

Fixation onsets come from exact sample matching of each word.rawEEG piece in sentenceData.rawData. Gaze
covariates come only from the matching word.rawET samples (timestamp ms, x, y, pupil). ZuCo's
sentence-level allFixations is not used: its entries do not line up with the word fixations (median
position error ~36 px, duration error ~110 ms under either index base). Fixations that are not on a word
therefore have no onset; long gaps between consecutive word fixations are kept as a separate event.
"""
import re

import h5py
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

FS_IN, DOWN = 500, 5
FS = FS_IN // DOWN
SOS = butter(4, 40, btype="low", fs=FS_IN, output="sos")
GAP_MS = 100.0          # inter-fixation interval above this is treated as a blink / off-word fixation gap


def subject_from_path(path):
    return re.sub(r"^results", "", str(path).rsplit("/", 1)[-1]).split("_")[0]


def _refs(ds):
    return ds[()].flatten()


def _is_empty(ds):
    if not isinstance(ds, h5py.Dataset):
        return False
    if ds.attrs.get("MATLAB_empty", 0) or ds.size == 0:
        return True
    if ds.dtype == np.uint64 and ds.shape == (2,):
        return True
    if np.issubdtype(ds.dtype, np.floating) and ds.size <= 1:
        return bool(np.all(np.isnan(ds[()])))
    return False


def _string(f, ref):
    return "".join(chr(int(c)) for c in np.asarray(f[ref][()]).flatten())


def _scalar(f, ref):
    ds = f[ref]
    if _is_empty(ds):
        return np.nan
    v = np.asarray(ds[()], dtype=float).flatten()
    return float(v[0]) if v.size else np.nan


def preprocess_sentence(raw):
    """raw (T500, C) float -> (x100 (T100, C) float32, bad100 (T100,) bool)."""
    bad = np.isnan(raw).any(1)
    mu = np.nanmean(raw, 0) if (~bad).any() else np.zeros(raw.shape[1])
    x = np.where(bad[:, None], 0.0, raw - mu)
    x = resample_poly(sosfiltfilt(SOS, x, axis=0), 1, DOWN, axis=0)
    T, nb = len(x), len(bad) // DOWN
    bad100 = np.zeros(T, bool)
    bad100[:nb] = bad[:nb * DOWN].reshape(nb, DOWN).any(1)
    bad100[nb:] = bad[nb * DOWN:].any()
    return x.astype(np.float32), bad100


def _gaze_trace(T500, et_pieces, offset_ms):
    """Interpolated gaze x/y at 500 Hz from per-fixation ET samples placed on the EEG timeline."""
    gx, gy = np.full(T500, np.nan), np.full(T500, np.nan)
    for et in et_pieces:
        idx = np.rint((et[:, 0] - offset_ms) / (1000.0 / FS_IN)).astype(int)
        ok = (idx >= 0) & (idx < T500) & (et[:, 1] > 0) & (et[:, 2] > 0)
        gx[idx[ok]], gy[idx[ok]] = et[ok, 1], et[ok, 2]
    have = np.isfinite(gx)
    if not have.any():
        return None
    t = np.arange(T500)
    gx = np.interp(t, t[have], gx[have])
    gy = np.interp(t, t[have], gy[have])
    return np.stack([gx, gy], 1)


def read_subject(path, sentences=None):
    """Read one subject file. Returns dict with per-sentence 100 Hz signals, fixations and word data."""
    out = []
    n_pieces = n_unmatched = 0
    with h5py.File(path, "r") as f:
        sd = f["sentenceData"]
        W, R, S = _refs(sd["word"]), _refs(sd["rawData"]), _refs(sd["content"])
        for i in range(len(W)):
            if sentences is not None and i not in sentences:
                continue
            w = f[W[i]]
            if not (isinstance(w, h5py.Group) and "rawEEG" in w and "rawET" in w and "content" in w):
                continue
            ds = f[R[i]]
            if _is_empty(ds) or ds.ndim != 2 or ds.shape[1] != 105:
                continue
            raw = np.asarray(ds[()], np.float64)
            lookup = {}
            for t in range(raw.shape[0]):
                lookup.setdefault(raw[t].tobytes(), t)
            words = [_string(f, r) for r in _refs(w["content"])]
            meas = {k: [_scalar(f, r) for r in _refs(w[k])] if k in w else [np.nan] * len(words)
                    for k in ("FFD", "GD", "TRT", "nFixations")}
            fixes, et_pieces, offsets = [], [], []
            E, Tr = _refs(w["rawEEG"]), _refs(w["rawET"])
            for j in range(len(words)):
                pe, pt = f[E[j]], f[Tr[j]]
                if not (isinstance(pe, h5py.Dataset) and pe.dtype == h5py.ref_dtype):
                    continue
                er = pe[()].flatten()
                tr = pt[()].flatten() if isinstance(pt, h5py.Dataset) and pt.dtype == h5py.ref_dtype else []
                for q, re_ in enumerate(er):
                    eeg = f[re_][()]
                    if eeg.ndim != 2 or eeg.shape[1] != 105 or eeg.shape[0] == 0:
                        continue
                    n_pieces += 1
                    t0 = lookup.get(np.asarray(eeg[0], np.float64).tobytes())
                    if t0 is None:
                        n_unmatched += 1
                        continue
                    et = f[tr[q]][()] if q < len(tr) else np.zeros((0, 4))
                    xy = np.full(2, np.nan)
                    if et.ndim == 2 and et.shape[1] >= 3 and len(et):
                        ok = (et[:, 1] > 0) & (et[:, 2] > 0)
                        if ok.any():
                            xy = et[ok, 1:3].mean(0)
                        et_pieces.append(et)
                        offsets.append(et[0, 0] - t0 * 1000.0 / FS_IN)
                    fixes.append((j, t0, eeg.shape[0], xy[0], xy[1]))
            x100, bad100 = preprocess_sentence(raw)
            gaze = None
            if offsets:
                g500 = _gaze_trace(raw.shape[0], et_pieces, float(np.median(offsets)))
                if g500 is not None:
                    gaze = resample_poly(g500 - g500.mean(0), 1, DOWN, axis=0)[:len(x100)] + g500.mean(0)
            bounds = np.asarray(f[_refs(sd["wordbounds"])[i]][()]) if "wordbounds" in sd else None
            out.append(dict(sent_idx=i, text=_string(f, S[i]), words=words, x=x100, bad=bad100, gaze=gaze,
                            fixes=fixes, meas=meas, bounds=bounds,
                            et_offset_sd=float(np.std(offsets)) if offsets else np.nan))
    return dict(subject=subject_from_path(path), sentences=out, n_pieces=n_pieces, n_unmatched=n_unmatched)

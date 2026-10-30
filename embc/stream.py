"""Concatenate one subject's sentences into a stream and derive the oculomotor event table."""
import numpy as np
import pandas as pd

from .io_zuco import DOWN, FS_IN, GAP_MS

MS = 1000.0 / FS_IN


def build_stream(subj):
    """subj: output of io_zuco.read_subject. Returns (Y, bad, seg_id, seg_table, gaze, fix_df, words)."""
    xs, bads, gazes, segs, rows, words = [], [], [], [], [], {}
    start = 0
    for k, s in enumerate(subj["sentences"]):
        T = len(s["x"])
        xs.append(s["x"]); bads.append(s["bad"])
        g = s["gaze"] if s["gaze"] is not None else np.full((T, 2), np.nan)
        gazes.append(g[:T])
        segs.append((s["sent_idx"], start, T))
        words[int(s["sent_idx"])] = s["words"]
        fx = sorted(s["fixes"], key=lambda r: r[1])
        first_seen = set()
        bounds = s["bounds"]
        for o, (wp, t500, n500, x, y) in enumerate(fx):
            prev = fx[o - 1] if o > 0 else None
            nxt = fx[o + 1] if o + 1 < len(fx) else None
            r = dict(subject=subj["subject"], sent_idx=s["sent_idx"], word_pos=wp, fix_ord=o,
                     onset=start + int(round(t500 / DOWN)), offset=start + int(round((t500 + n500) / DOWN)),
                     onset_ms=t500 * MS, dur_ms=n500 * MS, x=x, y=y, first_fix=wp not in first_seen,
                     word=s["words"][wp], n_words=len(s["words"]),
                     ffd=s["meas"]["FFD"][wp], gd=s["meas"]["GD"][wp], trt=s["meas"]["TRT"][wp],
                     nfix=s["meas"]["nFixations"][wp])
            first_seen.add(wp)
            if prev is not None:
                r.update(sacc_in_dx=x - prev[3], sacc_in_dy=y - prev[4],
                         gap_in_ms=(t500 - prev[1] - prev[2]) * MS)
            if nxt is not None:
                r.update(sacc_out_dx=nxt[3] - x, sacc_out_dy=nxt[4] - y,
                         gap_out_ms=(nxt[1] - t500 - n500) * MS)
            if bounds is not None and bounds.ndim == 2 and wp < bounds.shape[1]:
                r.update(wb_x1=bounds[0, wp], wb_x2=bounds[2, wp])
            rows.append(r)
        start += T
    Y = np.concatenate(xs)
    bad = np.concatenate(bads)
    gaze = np.concatenate(gazes)
    seg_table = np.array(segs, dtype=np.int64)
    seg_id = np.repeat(np.arange(len(segs)), seg_table[:, 2])
    fix = pd.DataFrame(rows)
    for c in ("sacc_in_dx", "sacc_in_dy", "gap_in_ms", "sacc_out_dx", "sacc_out_dy", "gap_out_ms", "wb_x1", "wb_x2"):
        if c not in fix:
            fix[c] = np.nan
    fix["sacc_in_amp"] = np.hypot(fix.sacc_in_dx, fix.sacc_in_dy)
    fix["sacc_out_amp"] = np.hypot(fix.sacc_out_dx, fix.sacc_out_dy)
    return Y, bad, seg_id, seg_table, gaze, fix, words


def _z(v):
    v = np.asarray(v, float)
    m, s = np.nanmean(v), np.nanstd(v)
    return np.nan_to_num((v - m) / (s if s > 0 else 1.0))


def design_inputs(fix, gaze, valid):
    """Oculomotor-only design inputs from the fixation table (no text columns are read)."""
    ocular = fix[["onset", "offset", "x", "y", "sacc_in_amp", "sacc_in_dx", "sacc_in_dy",
                  "sacc_out_amp", "sacc_out_dx", "sacc_out_dy", "gap_in_ms"]]
    events = {
        "fix": {"onset": ocular.onset.to_numpy(), "sacc_in_amp": _z(ocular.sacc_in_amp),
                "sacc_in_dx": _z(ocular.sacc_in_dx), "sacc_in_dy": _z(ocular.sacc_in_dy),
                "fix_x": _z(ocular.x), "fix_y": _z(ocular.y)},
        "sacc": {"onset": ocular.offset.to_numpy(), "sacc_out_amp": _z(ocular.sacc_out_amp),
                 "sacc_out_dx": _z(ocular.sacc_out_dx), "sacc_out_dy": _z(ocular.sacc_out_dy)},
    }
    long_gap = (ocular.gap_in_ms > GAP_MS).to_numpy()
    if long_gap.any():
        # gap starts where the previous fixation ended: onset minus the gap, in 100 Hz samples
        g_on = ocular.onset.to_numpy()[long_gap] - np.rint(ocular.gap_in_ms.to_numpy()[long_gap] / 10.0).astype(int)
        events["gap"] = {"onset": g_on}
    cont = {}
    for i, name in enumerate(("gaze_x", "gaze_y")):
        g = gaze[:, i].astype(float)
        ok = valid & np.isfinite(g)
        if ok.sum() > 10:
            m, s = g[ok].mean(), g[ok].std()
            cont[name] = np.where(np.isfinite(g), (g - m) / (s if s > 0 else 1.0), 0.0)
    return events, cont

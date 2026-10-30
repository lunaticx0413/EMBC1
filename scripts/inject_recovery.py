"""Stage 5: synthetic late-effect injection and recovery through residualization (doc section 19).

A known event-related signal s_k(t) = A_k h(t - t_k) * topography is injected at every eligible target
event (first fixation of a content word) of the continuous EEG. h is a smooth late wave (Gaussian peak
400 ms, SD 75 ms, support 0..800 ms) with a centro-parietal topography. A_k = 1 + 0.5 * (rho z(log FFD_k)
+ sqrt(1 - rho^2) eps_k): rho > 0 makes the injected amplitude collinear with fixation duration, the
case in which an oculomotor model could remove real lexical signal.

The residualization is linear in the data with a data-independent design, so for every condition
processed(EEG + c u topo) = processed(EEG) + c processed(u) topo exactly. processed(u) is computed once
per subject by refitting the same design, with the ridge recorded for that reader's stage-1a build, on the
injected time course u alone.

Reported per subject, condition (raw = no residualization, res, res_ovl), SNR and rho:
  signal path (noise-free): amplitude slope and correlation of recovered vs injected A_k, waveform NRMSE
                            and peak latency of the amplitude-weighted average recovered waveform
  noisy estimate:           correlation / slope of A_k estimated from processed injected epochs

Writes results/injection_recovery_fold<f>.csv and, next to it, a .json with the readers and the stage-1a
digests it was computed from; the report uses it for G0 only when it covers every reader of the current
stage-1a build.

    python scripts/inject_recovery.py --fold 1 [--subjects YAC,YAG] [--snr -10 -5 0 5] [--rho 0 0.5]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import events as ev  # noqa: E402
from embc import paths, runs  # noqa: E402
from embc.design import build_design, fit, intercept_kernel, predict_parts  # noqa: E402
from embc.inputs import LAGS, add_kernel, baseline, subject_epochs, subject_rng, window_index  # noqa: E402
from embc.stream import design_inputs  # noqa: E402

PEAK_MS, SD_MS, SUPPORT = 400.0, 75.0, 80          # support in 100 Hz samples (0..800 ms)
CONDS = ("raw", "res", "res_ovl")


def wave(lags):
    t = lags * 10.0
    return np.where(t >= 0, np.exp(-0.5 * ((t - PEAK_MS) / SD_MS) ** 2), 0.0)


def topography():
    """Unit-norm Gaussian (SD 4 cm) centred midway between Cz and Pz (E62): a centro-parietal late effect."""
    cp = pd.read_csv(paths.CHANPOS).set_index("name")
    centre = 0.5 * (cp.loc["Cz", ["x", "y", "z"]].to_numpy(float) + cp.loc["E62", ["x", "y", "z"]].to_numpy(float))
    xyz = cp[cp.is_constant == 0][["x", "y", "z"]].to_numpy(float)
    w = np.exp(-0.5 * (np.linalg.norm(xyz - centre, axis=1) / 4.0) ** 2)
    return w / np.linalg.norm(w)


def injected_course(n, seg_id, onsets, amps):
    u = np.zeros(n)
    L = np.arange(SUPPORT + 1)
    h = wave(L)
    for o, A in zip(onsets, amps):
        r = o + L
        ok = (r < n) & (seg_id[np.minimum(r, n - 1)] == seg_id[o])
        u[r[ok]] += A * h[ok]
    return u


def regress_slope(est, true):
    tc = true - true.mean()
    return float((tc * (est - est.mean())).sum() / (tc * tc).sum()), float(np.corrcoef(est, true)[0, 1])


def subject_recovery(s, E, snrs, rhos, seed):
    z = np.load(paths.continuous_path(s))
    if "gaze" not in z.files:
        raise SystemExit(f"{s}: continuous file has no gaze trace; rebuild with build_continuous.py --force")
    bad, seg, gaze = z["bad"], z["seg"], z["gaze"]
    n = len(bad)
    seg_id = np.repeat(np.arange(len(seg)), seg[:, 2])
    valid = ~bad
    fix = pd.read_csv(paths.DATA / "continuous" / f"{s}_fix.csv", keep_default_na=False, na_values=[""])
    ridge = json.loads(paths.subject_outputs(s)["info"].read_text())["ridge_rel"]   # as used for res/resfull
    events, cont = design_inputs(fix, gaze, valid)
    X, columns = build_design(n, seg_id, events, cont)
    tgt = E[(E.subject == s) & E.eligible.astype(bool)]
    onsets = tgt.onset.to_numpy()
    c_idx, wvalid = window_index(onsets, bad, seg_id)
    topo = topography()
    h = wave(LAGS)
    hb = baseline(h[None, :, None], np.ones((1, len(LAGS)), bool))[0, :, 0]
    eeg = {c: subject_epochs(s, onsets, c)[0] for c in CONDS}       # (n_ev, C, T)
    p_eeg = float(np.mean(eeg["raw"] ** 2))
    rng = subject_rng(s, seed)
    eps = rng.standard_normal(len(tgt))
    zffd = np.log(tgt.ffd_ms.to_numpy())
    zffd = (zffd - zffd.mean()) / zffd.std()
    rows = []
    for rho in rhos:
        A = 1.0 + 0.5 * (rho * zffd + np.sqrt(1 - rho ** 2) * eps)
        u = injected_course(n, seg_id, onsets, A)
        B, _ = fit(X, u[:, None], valid, ridge)
        kept, nuis = predict_parts(X, B, columns)
        res_u = u - nuis[:, 0]
        resfull_u = res_u - kept[:, 0]
        kl, K = intercept_kernel(B, columns)
        sig = {"raw": u[c_idx], "res": res_u[c_idx],
               "res_ovl": add_kernel(resfull_u[c_idx][:, :, None], kl, K)[:, :, 0]}
        sig = {k: baseline(v[:, :, None], wvalid)[:, :, 0] for k, v in sig.items()}   # (n_ev, T)
        p_sig1 = float(np.mean(sig["raw"] ** 2) * np.mean(topo ** 2))
        for cnd in CONDS:
            a_sig = (sig[cnd] * hb).sum(1) / (hb * hb).sum()
            sl, r = regress_slope(a_sig, A)
            w = (A[:, None] * sig[cnd]).sum(0) / (A * A).sum()
            nrmse = float(np.linalg.norm(w - hb) / np.linalg.norm(hb))
            lat = float(LAGS[np.argmax(w)] * 10.0)
            proj_eeg = np.einsum("nct,c,t->n", eeg[cnd], topo, hb) / (hb * hb).sum()
            for snr in snrs:
                cs = np.sqrt(10 ** (snr / 10.0) * p_eeg / p_sig1)
                a_noisy = proj_eeg / cs + a_sig
                nsl, nr = regress_slope(a_noisy, A)
                rows.append(dict(subject=s, rho=rho, cond=cnd, snr_db=snr, n_events=int(len(A)),
                                 sig_slope=sl, sig_corr=r, waveform_nrmse=nrmse, peak_ms=lat,
                                 noisy_slope=nsl, noisy_corr=nr))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--snr", type=float, nargs="+", default=[-10, -5, 0, 5])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    E = pd.read_csv(paths.fold_dir(a.fold) / "events.csv", keep_default_na=False, na_values=[""])
    subs = a.subjects.split(",") if a.subjects else ev.subjects_available()
    rows = []
    for s in subs:
        rows += subject_recovery(s, E, a.snr, a.rho, a.seed)
        print(s, "done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out or (paths.RESULTS / f"injection_recovery_fold{a.fold}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(fold=a.fold, seed=a.seed, subjects=sorted(subs),
                continuous={s: runs.continuous_digest(s) for s in sorted(subs)})
    out.with_suffix(".json").unlink(missing_ok=True)                 # the csv is valid only with its record
    runs.write_atomic(out, df.to_csv(index=False))
    runs.write_atomic(out.with_suffix(".json"), json.dumps(meta, indent=1))
    summ = df.groupby(["rho", "cond", "snr_db"])[["sig_slope", "sig_corr", "waveform_nrmse", "peak_ms", "noisy_slope",
                                                  "noisy_corr"]].mean().round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()

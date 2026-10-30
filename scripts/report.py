"""Stage 7: aggregate runs into tables and evaluate the pre-specified gates (PLAN.md section 6).

Per-event metrics are averaged over seeds, pooled over subject folds (disjoint test subjects), reduced to
subject means, and summarized with subject-bootstrap CIs. Contrasts are paired over subjects, on the events
both conditions cover; a gate uses a contrast only when both conditions cover the same events.

    python scripts/report.py --fold 1 [--exp main] [--n_perm 200]

Only completed runs are read, and only if their recorded inputs and code match the current ones (stale runs
are refused, not pooled) and every run of the experiment shares one protocol. Probes count only if computed
from the run's current embeddings; injection recovery counts for G0 only if it covers every reader of the
current stage-1a build.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import events as ev  # noqa: E402
from embc import lexicon as lx  # noqa: E402
from embc import metrics as mt  # noqa: E402
from embc import paths, runs  # noqa: E402

CONDS = ("raw", "res", "res_ovl", "ocfit", "raw_shift", "res_shift", "noise", "gaze", "prior")
METRICS = ("R@1", "R@5", "R@10", "RR", "nDCG")
PERM_GALLERIES = ("random_N100", "matched_N100", "balanced_exact_N100")
CONTRASTS = [("raw", "noise"), ("raw", "raw_shift"), ("res", "noise"), ("res", "res_shift"), ("res", "gaze"),
             ("res", "prior"), ("res", "ocfit"), ("raw", "res"), ("res", "res_ovl"), ("res_ovl", "res_shift")]


def completed_runs(base):
    """Completed runs of one experiment, checked against the current inputs and code. Refuses stale runs,
    runs of different protocols and an experiment without completed runs instead of reporting them."""
    ok, stale = [], []
    for d in sorted(base.glob("sf*/*/seed*")):
        if not (d / "metrics.json").exists():
            continue
        why = runs.stale_reason(d)
        if why:
            stale.append(f"{d}: {why}")
        else:
            ok.append(d)
    if stale:
        raise SystemExit("refusing to report stale runs (retrain them, e.g. OVERWRITE=1 bash scripts/run_mvp.sh, "
                         "or report another --exp):\n  " + "\n  ".join(stale))
    if not ok:
        raise SystemExit(f"no completed runs under {base}; nothing written")
    protos = {json.dumps(runs.protocol(runs.read_metrics(d)["config"]), sort_keys=True) for d in ok}
    if len(protos) > 1:
        raise SystemExit(f"runs under {base} mix {len(protos)} protocols (arguments other than reader fold, "
                         "condition and seed differ); use one --exp per protocol:\n  " + "\n  ".join(sorted(protos)))
    return set(ok)


def per_event(base, cond, gallery, ok):
    """Seed-averaged per-event metrics pooled over subject folds, or None if no completed run exists."""
    frames = []
    for sf in sorted(base.glob("sf*")):
        seeds = [pd.read_csv(d / f"test_{gallery}.csv") for d in sorted((sf / cond).glob("seed*"))
                 if d in ok and (d / f"test_{gallery}.csv").exists()]
        if seeds:
            d = pd.concat(seeds).groupby(["event_id", "subject"], as_index=False)[list(METRICS)].mean()
            d["n_seeds"] = len(seeds)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize(d):
    out = {"n_events": int(len(d)), "n_subjects": int(d.subject.nunique()), "n_seeds": int(d.n_seeds.min())}
    for k in METRICS:
        _, sm = mt.subject_means(d[k].to_numpy(), d.subject.to_numpy())
        m, lo, hi = mt.bootstrap_ci(sm)
        out[k] = dict(mean=m, lo=lo, hi=hi)
    return out


def contrast(da, db):
    """Paired MRR contrast on the events both conditions cover, or None if they share none. Within a reader
    fold every condition has the same test events, so the sets differ only when a condition lacks reader
    folds; complete is False then."""
    m = da.merge(db, on=["event_id", "subject"], suffixes=("_a", "_b"))
    if m.empty:
        return None
    c = mt.paired_contrast(m["RR_a"].to_numpy(), m["RR_b"].to_numpy(), m.subject.to_numpy())
    return dict(c, n_events=int(len(m)), complete=bool(len(m) == len(da) == len(db)))


def permutation_p(base, fold, cond, gallery, n_perm, ok, seed=0):
    """Within-subject permutation null of subject-mean MRR for seed-0 runs, pooled over subject folds."""
    fdir = paths.fold_dir(fold)
    T = np.load(fdir / "types.npz")["emb"].astype(np.float32)
    obs, null = [], []
    for sf in sorted(base.glob("sf*")):
        run = sf / cond / f"seed{seed}"
        if run not in ok:
            return None
        z = np.load(run / "z.npz")
        r, G = lx.load_galleries(fdir, z["test_ids"])[gallery]
        ids = z["test_ids"][r]
        df = pd.read_csv(run / f"test_{gallery}.csv").set_index("event_id").loc[ids]
        subj = df.subject.to_numpy()
        _, sm = mt.subject_means(df.RR.to_numpy(), subj)
        obs.append(sm)
        null.append(mt.permutation_null(z["z_test"].astype(np.float32)[r], T, G, subj, n_perm=n_perm, seed=seed))
    o = np.concatenate(obs).mean()
    nl = np.concatenate(null, axis=1).mean(1)
    return dict(observed=float(o), null_mean=float(nl.mean()), null_q95=float(np.quantile(nl, 0.95)),
                p=float((1 + (nl >= o).sum()) / (1 + len(nl))), n_perm=n_perm)


def probes_table(ok, notes):
    """Probe R^2 per condition and target over the runs whose probes were computed from their current z.npz."""
    rows = []
    for d in sorted(ok):
        f = d / "probes.json"
        if not f.exists():
            continue
        p = json.loads(f.read_text())
        if p.get("source", {}).get("z") != runs.file_digest(d / "z.npz"):
            notes.append(f"probes of {d} ignored: not computed from its current embeddings (rerun the probes stage)")
            continue
        for tgt, r in p["targets"].items():
            rows.append(dict(cond=d.parent.name, target=tgt, linear_r2=r["linear"]["r2"], mlp_r2=r["mlp"]["r2"]))
    if not rows:
        return None
    return pd.DataFrame(rows).groupby(["cond", "target"], as_index=False)[["linear_r2", "mlp_r2"]].mean()


def injection(fold, notes):
    """Injection-recovery rows of this text fold, or None unless they were computed from every reader of the
    current stage-1a build (a smoke run on two readers or a rebuilt stage 1a does not count)."""
    f = paths.RESULTS / f"injection_recovery_fold{fold}.csv"
    meta_f = f.with_suffix(".json")
    if not (f.exists() and meta_f.exists()):
        notes.append(f"no injection recovery for fold {fold} ({f.name} with its .json record); G0 not evaluated")
        return None
    meta = json.loads(meta_f.read_text())
    subs = ev.subjects_available()
    if meta.get("subjects") != subs:
        notes.append(f"injection recovery covers {len(meta.get('subjects', []))} of {len(subs)} readers; G0 not evaluated")
        return None
    if meta.get("continuous") != {s: runs.continuous_digest(s) for s in subs}:
        notes.append("injection recovery was computed from another stage-1a build; G0 not evaluated")
        return None
    return pd.read_csv(f)


def gates(S, C, perm, probes, inj):
    def lo_pos(a, b, g):
        c = C.get(g, {}).get(f"{a}-{b}")
        return None if c is None or not c["complete"] else bool(c["lo"] > 0)

    chance_rr = mt.chance(100)["RR"]
    g = {}
    g["G0_injection_preserved"] = None
    if inj is not None:
        x = inj[(inj.rho == 0) & inj.cond.isin(["res", "res_ovl"])].groupby("cond").sig_slope.mean()
        g["G0_injection_preserved"] = bool((x >= 0.8).all()) if len(x) == 2 else None
    g["G1_raw_beats_noise"] = lo_pos("raw", "noise", "random_N100")
    g["G1_raw_beats_shift"] = lo_pos("raw", "raw_shift", "random_N100")
    p = perm.get("raw/random_N100")
    g["G1_raw_permutation"] = None if p is None else bool(p["p"] < 0.05)
    # chance is only a meaningful floor where the training-frequency prior is neutral (balanced_exact)
    res_b = S.get("balanced_exact_N100", {}).get("res")
    g["G2_res_above_chance_balanced_exact"] = None if res_b is None else bool(res_b["RR"]["lo"] > chance_rr)
    g["G2_res_beats_noise_balanced_exact"] = lo_pos("res", "noise", "balanced_exact_N100")
    for b in ("res_shift", "gaze", "prior", "noise"):
        g[f"G2_res_beats_{b}_matched"] = lo_pos("res", b, "matched_N100")
    if probes is not None and {"raw", "res"} <= set(probes.cond):
        pv = probes.pivot(index="target", columns="cond", values="linear_r2")
        g["G3_probe_decrease_fraction"] = float((pv["res"] < pv["raw"]).mean())
        g["G3_probe_decrease"] = bool(g["G3_probe_decrease_fraction"] >= 4 / 7)
    else:
        g["G3_probe_decrease"] = None
    vals = [g[k] for k in ("G1_raw_beats_noise", "G1_raw_beats_shift", "G1_raw_permutation")]
    g["G1"] = None if None in vals else all(vals)
    vals = [g[k] for k in g if k.startswith("G2_")]
    g["G2"] = None if None in vals else all(vals)
    g["primary_criterion_doc_s32"] = (None if None in (g["G2"], g["G0_injection_preserved"], g["G3_probe_decrease"])
                                      else g["G2"] and g["G0_injection_preserved"] and g["G3_probe_decrease"])
    return g


def fmt(s, k="RR"):
    return f"{s[k]['mean']:.4f} [{s[k]['lo']:.4f}, {s[k]['hi']:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--exp", default="main")
    a = ap.parse_args()
    base = paths.RUNS / a.exp / f"fold{a.fold}"
    tag = f"fold{a.fold}" if a.exp == "main" else f"{a.exp}_fold{a.fold}"
    ok = completed_runs(base)
    notes = []
    stems = sorted(p.stem.replace("galleries_", "") for p in paths.fold_dir(a.fold).glob("galleries_*.npz"))
    galleries = stems + [g.replace("balanced_", "balanced_exact_") for g in stems if g.startswith("balanced_")]
    S, C, D, perm = {}, {}, {}, {}
    for g in galleries:
        D[g] = {c: d for c in CONDS if (d := per_event(base, c, g, ok)) is not None}
        S[g] = {c: summarize(d) for c, d in D[g].items()}
        C[g] = {f"{x}-{y}": c for x, y in CONTRASTS
                if x in D[g] and y in D[g] and (c := contrast(D[g][x], D[g][y])) is not None}
        if a.n_perm:
            for c in ("raw", "res"):
                if g in PERM_GALLERIES and c in D[g] and (r := permutation_p(base, a.fold, c, g, a.n_perm, ok)):
                    perm[f"{c}/{g}"] = r
    probes = probes_table(ok, notes)
    inj = injection(a.fold, notes)
    G = gates(S, C, perm, probes, inj)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out = dict(fold=a.fold, exp=a.exp, n_runs=len(ok), summary=S, contrasts=C, permutation=perm, gates=G,
               notes=notes)
    (paths.RESULTS / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))

    L = [f"# Results, experiment {a.exp}, text fold {a.fold}", "", "Generated by `scripts/report.py`; do not edit by hand.", "",
         "Subject-level mean [95% subject-bootstrap CI]; per-event metrics averaged over seeds; "
         "unseen subjects x unseen text block.", ""]
    for g in galleries:
        N = int(g.split("_N")[1])
        ch = mt.chance(N)
        L += [f"## Gallery {g}", "", "| condition | events | subjects | seeds | R@1 | R@5 | R@10 | MRR | nDCG |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c, s in S[g].items():
            L.append(f"| {c} | {s['n_events']} | {s['n_subjects']} | {s['n_seeds']} | "
                     + " | ".join(f"{s[k]['mean']:.4f}" for k in METRICS) + " |")
        L.append("| chance | | | | " + " | ".join(f"{ch[k]:.4f}" for k in METRICS) + " |")
        L += ["", "MRR with CI: " + "; ".join(f"{c} {fmt(s)}" for c, s in S[g].items()), "",
              "| contrast (MRR) | diff | 95% CI | sign-flip p | subjects | events |", "|---|---:|---|---:|---:|---:|"]
        for k, v in C[g].items():
            ev_n = f"{v['n_events']}" + ("" if v["complete"] else " (shared only)")
            L.append(f"| {k} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['p_signflip']:.4f} | "
                     f"{v['n_subjects']} | {ev_n} |")
        L.append("")
    if perm:
        L += ["## Within-subject permutation null (seed 0)", "", "| run | observed MRR | null mean | null q95 | p |",
              "|---|---:|---:|---:|---:|"]
        L += [f"| {k} | {v['observed']:.4f} | {v['null_mean']:.4f} | {v['null_q95']:.4f} | {v['p']:.4f} |"
              for k, v in perm.items()]
        L.append("")
    if probes is not None:
        L += ["## Post-hoc oculomotor probes (test R^2, mean over runs)", "", probes.round(4).to_markdown(index=False), ""]
    if inj is not None:
        t = inj.groupby(["rho", "cond", "snr_db"])[["sig_slope", "sig_corr", "waveform_nrmse", "peak_ms",
                                                   "noisy_corr"]].mean().round(3).reset_index()
        L += ["## Synthetic injection recovery (mean over subjects)", "", t.to_markdown(index=False), ""]
    L += ["## Gates (PLAN.md section 6)", "", "| gate | status |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in G.items()]
    if notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    (paths.RESULTS / f"TABLES_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

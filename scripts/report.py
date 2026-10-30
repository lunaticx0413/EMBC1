"""Stage 7: aggregate runs into tables and evaluate the pre-specified gates (PLAN.md section 6).

Per-event metrics are averaged over seeds, pooled over subject folds (disjoint test subjects), reduced to
subject means, and summarized with subject-bootstrap CIs. Contrasts are paired over subjects.

    python scripts/report.py --fold 1 [--n_perm 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import lexicon as lx  # noqa: E402
from embc import metrics as mt  # noqa: E402
from embc import paths  # noqa: E402

CONDS = ("raw", "res", "res_ovl", "ocfit", "raw_shift", "res_shift", "noise", "gaze", "prior")
METRICS = ("R@1", "R@5", "R@10", "RR", "nDCG")
PERM_GALLERIES = ("random_N100", "matched_N100", "balanced_exact_N100")
CONTRASTS = [("raw", "noise"), ("raw", "raw_shift"), ("res", "noise"), ("res", "res_shift"), ("res", "gaze"),
             ("res", "prior"), ("res", "ocfit"), ("raw", "res"), ("res", "res_ovl"), ("res_ovl", "res_shift")]


def per_event(fold, cond, gallery):
    """Seed-averaged per-event metrics pooled over subject folds, or None if no run exists."""
    base = paths.RUNS / f"fold{fold}"
    frames = []
    for sf in sorted(base.glob("sf*")):
        seeds = [pd.read_csv(f) for f in sorted((sf / cond).glob(f"seed*/test_{gallery}.csv"))]
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
    m = da.merge(db, on=["event_id", "subject"], suffixes=("_a", "_b"))
    if len(m) != len(da) or len(m) != len(db):
        raise ValueError("contrast over non-identical event sets")
    return mt.paired_contrast(m["RR_a"].to_numpy(), m["RR_b"].to_numpy(), m.subject.to_numpy())


def permutation_p(fold, cond, gallery, n_perm, seed=0):
    """Within-subject permutation null of subject-mean MRR for seed-0 runs, pooled over subject folds."""
    fdir = paths.fold_dir(fold)
    T = np.load(fdir / "types.npz")["emb"].astype(np.float32)
    obs, null = [], []
    for sf in sorted((paths.RUNS / f"fold{fold}").glob("sf*")):
        run = sf / cond / f"seed{seed}"
        if not (run / "z.npz").exists():
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


def probes_table(fold):
    rows = []
    for f in (paths.RUNS / f"fold{fold}").glob("sf*/*/seed*/probes.json"):
        cond = f.parent.parent.name
        for tgt, r in json.loads(f.read_text()).items():
            rows.append(dict(cond=cond, target=tgt, linear_r2=r["linear"]["r2"], mlp_r2=r["mlp"]["r2"]))
    if not rows:
        return None
    return pd.DataFrame(rows).groupby(["cond", "target"], as_index=False)[["linear_r2", "mlp_r2"]].mean()


def gates(S, C, perm, probes, inj):
    def lo_pos(a, b, g):
        c = C.get(g, {}).get(f"{a}-{b}")
        return None if c is None else bool(c["lo"] > 0)

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
    a = ap.parse_args()
    stems = sorted(p.stem.replace("galleries_", "") for p in paths.fold_dir(a.fold).glob("galleries_*.npz"))
    galleries = stems + [g.replace("balanced_", "balanced_exact_") for g in stems if g.startswith("balanced_")]
    S, C, D, perm = {}, {}, {}, {}
    for g in galleries:
        D[g] = {c: d for c in CONDS if (d := per_event(a.fold, c, g)) is not None}
        S[g] = {c: summarize(d) for c, d in D[g].items()}
        C[g] = {f"{x}-{y}": contrast(D[g][x], D[g][y]) for x, y in CONTRASTS if x in D[g] and y in D[g]}
        if a.n_perm:
            for c in ("raw", "res"):
                if g in PERM_GALLERIES and c in D[g] and (r := permutation_p(a.fold, c, g, a.n_perm)) is not None:
                    perm[f"{c}/{g}"] = r
    probes = probes_table(a.fold)
    inj_f = paths.RESULTS / "injection_recovery.csv"
    inj = pd.read_csv(inj_f) if inj_f.exists() else None
    G = gates(S, C, perm, probes, inj)
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out = dict(fold=a.fold, summary=S, contrasts=C, permutation=perm, gates=G)
    (paths.RESULTS / f"summary_fold{a.fold}.json").write_text(json.dumps(out, indent=1))

    L = [f"# Results, text fold {a.fold}", "", "Generated by `scripts/report.py`; do not edit by hand.", "",
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
              "| contrast (MRR) | diff | 95% CI | sign-flip p |", "|---|---:|---|---:|"]
        for k, v in C[g].items():
            L.append(f"| {k} | {v['diff']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['p_signflip']:.4f} |")
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
    (paths.RESULTS / f"TABLES_fold{a.fold}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

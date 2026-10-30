"""Stage 4: post-hoc oculomotor probes on frozen encoder embeddings (the probes never train the encoder).

Linear (ridge, CV alpha) and MLP probes fitted on training-partition embeddings and scored on the test
partition (held-out subjects and texts). Blink state is not probed: ZuCo ships no blink annotations; long
inter-fixation gaps are only a proxy and are part of the residualization design.

Only completed runs whose inputs and code are current are probed. probes.json records the digest of the
z.npz it was computed from, so the report can drop probes of a run that has since been retrained.

    python scripts/probes.py --runs runs/main/fold1/sf0/raw/seed0 runs/main/fold1/sf0/res/seed0
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import paths, runs  # noqa: E402

TARGETS = dict(log_ffd=lambda e: np.log(e.ffd_ms), log_gd=lambda e: np.log(e.gd_ms),
               sacc_in_amp=lambda e: e.sacc_in_amp, sacc_out_amp=lambda e: e.sacc_out_amp,
               sacc_in_dx=lambda e: e.sacc_in_dx, fix_x=lambda e: e.x, fix_y=lambda e: e.y)


def probe_run(run, E, max_train=20000, seed=0):
    z = np.load(run / "z.npz")
    idx = pd.Series(np.arange(len(E)), index=E.event_id)
    Etr, Ete = E.iloc[idx.loc[z["train_ids"]].to_numpy()], E.iloc[idx.loc[z["test_ids"]].to_numpy()]
    Ztr, Zte = z["z_train"].astype(np.float32), z["z_test"].astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(Ztr) > max_train:
        keep = np.sort(rng.choice(len(Ztr), max_train, replace=False))
        Ztr, Etr = Ztr[keep], Etr.iloc[keep]
    sc = StandardScaler().fit(Ztr)
    Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
    out = {}
    for name, f in TARGETS.items():
        ytr, yte = np.asarray(f(Etr), float), np.asarray(f(Ete), float)
        a, b = np.isfinite(ytr), np.isfinite(yte)
        mu, sd = ytr[a].mean(), ytr[a].std()
        res = {}
        lin = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Ztr[a], (ytr[a] - mu) / sd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300,
                               random_state=seed).fit(Ztr[a], (ytr[a] - mu) / sd)
        for kind, m in (("linear", lin), ("mlp", mlp)):
            pred = m.predict(Zte[b]) * sd + mu
            res[kind] = dict(r2=float(r2_score(yte[b], pred)), mae=float(mean_absolute_error(yte[b], pred)))
        res["n_train"], res["n_test"] = int(a.sum()), int(b.sum())
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    events = {}
    for r in a.runs:
        r = Path(r)
        why = runs.stale_reason(r)
        if why:
            print(f"{r}: {why}, skipped", flush=True)
            continue
        fold = runs.read_metrics(r)["config"]["fold"]                # the events the run was trained on
        if fold not in events:
            events[fold] = pd.read_csv(paths.fold_dir(fold) / "events.csv", keep_default_na=False, na_values=[""])
        res = probe_run(r, events[fold])
        runs.write_atomic(r / "probes.json", json.dumps(dict(source=dict(z=runs.file_digest(r / "z.npz")),
                                                             targets=res), indent=1))
        print(r, json.dumps({k: round(v["linear"]["r2"], 3) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()

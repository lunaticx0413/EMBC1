"""Stage 6: collinearity of oculomotor and lexical variables (doc section 6) and real-data validity of the
residualization (doc section 20): variance retention, grand-average FRP preservation, split-half reliability.

Surprisal is read, for diagnostics only, from the parent project's GPT-2 surprisal cache when present.
Outputs are per text fold: results/collinearity_fold<f>.json, validity_per_subject_fold<f>.csv,
validity_summary_fold<f>.json and data/fold<f>/grand_average_frp.npz.

    python scripts/diagnostics.py --fold 1
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
from embc import paths  # noqa: E402
from embc.inputs import LAGS, subject_epochs  # noqa: E402

PARENT = Path("/run/media/lunaticx/Documents/eeg2text/data/zuco2_NR_GD")
KEY_PAIRS = [("nchar", "sacc_in_amp"), ("nchar", "sacc_out_amp"), ("nchar", "landing"), ("nchar", "fix_x"),
             ("zipf", "log_ffd"), ("surprisal", "log_gd"), ("surprisal", "log_trt"), ("log_ffd", "sacc_in_amp"),
             ("log_ffd", "sacc_out_amp")]


def attach_surprisal(E, index_csv, surprisal_json):
    if not (index_csv.exists() and surprisal_json.exists()):
        return E.assign(surprisal=np.nan)
    idx = pd.read_csv(index_csv, keep_default_na=False, dtype={"subject": str, "text_key": str})
    key = dict(zip(zip(idx.subject, idx.sent_idx.astype(int)), idx.text_key))
    sur = json.loads(surprisal_json.read_text())
    vals = []
    for s, i, w, n in zip(E.subject, E.sent_idx, E.word_pos, E.n_words):
        lst = sur.get(key.get((s, int(i)), ""), None)
        vals.append(lst[w] if lst is not None and len(lst) == n and w < len(lst) else np.nan)
    return E.assign(surprisal=vals)


def vif(df):
    C = np.corrcoef(df.to_numpy(float), rowvar=False)
    return dict(zip(df.columns, np.diag(np.linalg.pinv(C)).round(3).tolist()))


def collinearity(E):
    V = pd.DataFrame(dict(nchar=E.nchar, zipf=E.zipf, surprisal=E.surprisal, log_ffd=np.log(E.ffd_ms),
                          log_gd=np.log(E.gd_ms), log_trt=np.log(E.trt_ms), sacc_in_amp=E.sacc_in_amp,
                          sacc_out_amp=E.sacc_out_amp, fix_x=E.x,
                          landing=(E.x - E.wb_x1) / (E.wb_x2 - E.wb_x1)))
    V = V.replace([np.inf, -np.inf], np.nan)
    if V.surprisal.isna().all():
        V = V.drop(columns="surprisal")
    cc = V.dropna()
    pairs = [dict(a=x, b=y, pearson=float(cc[x].corr(cc[y])), spearman=float(cc[x].corr(cc[y], method="spearman")))
             for x, y in KEY_PAIRS if x in cc and y in cc]
    return dict(n_complete=int(len(cc)), pearson=cc.corr().round(3).to_dict(),
                spearman=cc.corr(method="spearman").round(3).to_dict(), vif=vif(cc), key_pairs=pairs)


def split_half(X, rng):
    perm = rng.permutation(len(X))
    a, b = X[perm[::2]].mean(0), X[perm[1::2]].mean(0)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def validity(E, subjects, conds=("raw", "res", "res_ovl"), seed=0):
    rng = np.random.default_rng(seed)
    per, ga = [], {c: [] for c in conds}
    for s in subjects:
        info = json.loads((paths.DATA / "continuous" / f"{s}_info.json").read_text())
        t = E[(E.subject == s) & E.eligible.astype(bool)]
        row = dict(subject=s, var_res_over_raw=info["var_res"] / info["var_raw"],
                   var_resfull_over_raw=info["var_resfull"] / info["var_raw"],
                   var_nuisance_over_raw=info["var_nuisance"] / info["var_raw"], cond_design=info["cond_design"],
                   n_events=int(len(t)))
        avg = {}
        for c in conds:
            X = subject_epochs(s, t.onset.to_numpy(), c)[0]
            avg[c] = X.mean(0)
            ga[c].append(avg[c])
            row[f"split_half_{c}"] = split_half(X, rng)
            row[f"epoch_var_{c}"] = float(X.var(0).mean())
        for c in conds[1:]:
            row[f"frp_corr_raw_{c}"] = float(np.corrcoef(avg["raw"].ravel(), avg[c].ravel())[0, 1])
            row[f"frp_gfp_ratio_{c}"] = float(avg[c].std(0).mean() / avg["raw"].std(0).mean())
        per.append(row)
        print(s, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    return pd.DataFrame(per), {c: np.stack(v) for c, v in ga.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--subjects", default=None)
    a = ap.parse_args()
    E = pd.read_csv(paths.fold_dir(a.fold) / "events.csv", keep_default_na=False, na_values=[""])
    E = attach_surprisal(E, PARENT / "index.csv", PARENT / "surprisal_gpt2.json")
    subs = a.subjects.split(",") if a.subjects else ev.subjects_available()
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    col = collinearity(E[E.eligible.astype(bool)])
    (paths.RESULTS / f"collinearity_fold{a.fold}.json").write_text(json.dumps(col, indent=1))
    print(json.dumps(col["key_pairs"], indent=1))
    print("VIF", col["vif"])
    per, ga = validity(E, subs)
    per.to_csv(paths.RESULTS / f"validity_per_subject_fold{a.fold}.csv", index=False)
    np.savez(paths.DATA / f"fold{a.fold}" / "grand_average_frp.npz", lags_ms=LAGS * 10, subjects=np.array(subs),
             **{c: v.astype(np.float32) for c, v in ga.items()})
    summ = per.drop(columns="subject").mean().round(4).to_dict()
    (paths.RESULTS / f"validity_summary_fold{a.fold}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

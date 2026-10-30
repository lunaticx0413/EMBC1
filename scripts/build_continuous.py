"""Stage 1: per-subject continuous stream, fixation table and oculomotor residualization.

Writes data/continuous/<SUBJ>.npz (raw, res, resfull as float16; bad mask; sentence segments; gaze trace;
fixation intercept kernel), <SUBJ>_fix.csv (all word fixations), <SUBJ>_words.json and <SUBJ>_info.json.
Resumable: finished subjects are skipped.

    python scripts/build_continuous.py [--subjects YAC,YAG] [--ridge 1e-3]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import paths  # noqa: E402
from embc.design import residualize  # noqa: E402
from embc.io_zuco import read_subject, subject_from_path  # noqa: E402
from embc.stream import build_stream, design_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max_sentences", type=int, default=None, help="smoke tests only")
    ap.add_argument("--force", action="store_true", help="rebuild finished subjects")
    a = ap.parse_args()
    out = paths.DATA / "continuous"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(paths.ZUCO2_NR / "*_NR.mat")))
    if a.subjects:
        keep = set(a.subjects.split(","))
        files = [p for p in files if subject_from_path(p) in keep]
    for p in files:
        s = subject_from_path(p)
        if paths.continuous_path(s).exists() and not a.force:
            print(f"{s}: done, skipping", flush=True)
            continue
        t0 = time.time()
        sents = set(range(a.max_sentences)) if a.max_sentences else None
        subj = read_subject(p, sents)
        Y, bad, seg_id, seg, gaze, fix, words = build_stream(subj)
        valid = ~bad
        events, cont = design_inputs(fix, gaze, valid)
        r = residualize(Y, valid, seg_id, events, cont, ridge=a.ridge)
        info = dict(subject=s, n_sentences=len(seg), n_samples=int(len(Y)), n_fixations=int(len(fix)),
                    n_pieces=subj["n_pieces"], n_unmatched=subj["n_unmatched"],
                    n_gap_events=int(len(events.get("gap", {}).get("onset", []))),
                    gaze_coverage=float(np.isfinite(gaze[:, 0])[valid].mean()),
                    et_offset_sd_ms_max=float(np.nanmax([x["et_offset_sd"] for x in subj["sentences"]])),
                    ridge_rel=a.ridge, **r["info"], seconds=round(time.time() - t0, 1))
        tmp = str(paths.continuous_path(s)) + ".tmp.npz"
        np.savez(tmp, raw=Y.astype(np.float16), res=r["res"].astype(np.float16),
                 resfull=r["resfull"].astype(np.float16), bad=bad, seg=seg, gaze=gaze.astype(np.float32),
                 kernel=r["kernel"].astype(np.float32), kernel_lags=r["kernel_lags"])
        os.replace(tmp, paths.continuous_path(s))
        fix.to_csv(out / f"{s}_fix.csv", index=False)
        with open(out / f"{s}_words.json", "w") as fh:
            json.dump({str(k): v for k, v in words.items()}, fh)
        with open(out / f"{s}_info.json", "w") as fh:
            json.dump(info, fh, indent=1)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
